from transformers import (
    PreTrainedModel, 
    WhisperPreTrainedModel
)

from transformers.models.whisper.modeling_whisper import (
    WhisperEncoderLayer,
    WHISPER_ATTENTION_CLASSES,
    ACT2FN,
    WhisperAttention,
    WhisperPositionalEmbedding,
    _prepare_4d_causal_attention_mask,
    BaseModelOutputWithPastAndCrossAttentions
)
from transformers.modeling_outputs import Seq2SeqLMOutput
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask
from typing import Optional, Tuple
import math
from torch import nn
from src.modeling_sa_config import ConditionalSpeakerGenerationConfig
import torch
from transformers.utils import logging
import random

logger = logging.get_logger(__name__)

class CrossAttention(WhisperAttention):
    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[Tuple[torch.Tensor]] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None

        bsz, tgt_len, _ = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling
        # get key, value proj
        # `past_key_value[0].shape[2] == key_value_states.shape[1]`
        # is checking that the `sequence_length` of the `past_key_value` is the same as
        # the provided `key_value_states` to support prefix tuning
        if (
            is_cross_attention
            and past_key_value is not None
            and past_key_value[0].shape[2] == key_value_states[0].shape[1]
        ):
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            # cross_attentions
            key_states = self._shape(self.k_proj(key_value_states[0]), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states[1]), -1, bsz)
        elif past_key_value is not None:
            # reuse k, v, self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            # self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_states, value_states)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.reshape(*proj_shape)
        value_states = value_states.reshape(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if layer_head_mask is not None:
            if layer_head_mask.size() != (self.num_heads,):
                raise ValueError(
                    f"Head mask for a single layer should be of size {(self.num_heads,)}, but is"
                    f" {layer_head_mask.size()}"
                )
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if output_attentions:
            # this operation is a bit awkward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to be reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz * self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)

        # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
        # partitioned across GPUs when using tensor-parallelism.
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped, past_key_value

class SpeakerDecoderLayer(nn.Module):
    def __init__(self, config: ConditionalSpeakerGenerationConfig, layer_idx: int = 0):
        super().__init__()
        self.embed_dim = config.d_model
        self.layer_idx = layer_idx
        self.self_spk_attn = WHISPER_ATTENTION_CLASSES[config._attn_implementation](
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
            is_causal=True,
            config=config,
        )
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.speaker_self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.speaker_encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.speaker_encoder_attn = CrossAttention(
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
            config=config,
        )        
        self.speaker_fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        self.speaker_fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)
        self.speaker_final_layer_norm = nn.LayerNorm(self.embed_dim)
        self.speaker_decoder_layers_frozen = config.speaker_decoder_layers_frozen

    def forward(
        self,
        # hidden_states: torch.Tensor,
        speaker_hidden_states: torch.Tensor,
        
        attention_mask: Optional[torch.Tensor] = None,
        
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_speaker_features: Optional[torch.Tensor] = None,
        
        encoder_attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        cross_attn_layer_head_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = True,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            encoder_hidden_states (`torch.FloatTensor`):
                cross attention input to the layer of shape `(batch, seq_len, embed_dim)`
            encoder_attention_mask (`torch.FloatTensor`): encoder attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (`torch.FloatTensor`): mask for attention heads in a given layer of size
                `(encoder_attention_heads,)`.
            cross_attn_layer_head_mask (`torch.FloatTensor`): mask for cross-attention heads in a given layer of
                size `(decoder_attention_heads,)`.
            past_key_value (`Tuple(torch.FloatTensor)`): cached past key and value projection states
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        
        ###############################SPEAKER PART##############################################
        speaker_residual = speaker_hidden_states
        speaker_hidden_states = self.speaker_self_attn_layer_norm(speaker_hidden_states)

        # Self Attention
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_speaker_attn_past_key_value = past_key_value[1][:2] if past_key_value is not None else None
        # add present self-attn cache to positions 1,2 of present_key_value tuple
        speaker_hidden_states, self_speaker_attn_weights, speaker_present_key_value = self.self_spk_attn(
            hidden_states=speaker_hidden_states,
            past_key_value=self_speaker_attn_past_key_value,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        speaker_hidden_states = nn.functional.dropout(speaker_hidden_states, p=self.dropout, training=self.training)
        speaker_hidden_states = speaker_residual + speaker_hidden_states

        # Cross Attention Block for speaker features
        speaker_cross_attn_present_key_value = None
        speaker_cross_attn_weights = None
        if encoder_speaker_features is not None:
            speaker_residual = speaker_hidden_states
            speaker_hidden_states = self.speaker_encoder_attn_layer_norm(speaker_hidden_states)
            # cross_attn cached key/values tuple is at positions 3,4 of present_key_value tuple
            speaker_cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            
            if self.layer_idx < self.speaker_decoder_layers_frozen - 1:
                key_value_states = (encoder_hidden_states, encoder_hidden_states)
                encoder_attention_mask = None
            elif self.layer_idx == self.speaker_decoder_layers_frozen - 1:
                key_value_states = (encoder_hidden_states, encoder_speaker_features)
            else:
                key_value_states = (encoder_speaker_features, encoder_speaker_features)
                
            speaker_hidden_states, speaker_cross_attn_weights, speaker_cross_attn_present_key_value = self.speaker_encoder_attn(
                hidden_states=speaker_hidden_states,
                key_value_states=key_value_states,
                attention_mask=encoder_attention_mask,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=speaker_cross_attn_past_key_value,
                output_attentions=output_attentions,
            )
            speaker_hidden_states = nn.functional.dropout(speaker_hidden_states, p=self.dropout, training=self.training)
            # if self.layer_idx < self.speaker_decoder_layers_frozen - 1:
            #     # For the first layer, speaker_residual is the word embedding
            #     # speaker_hidden_states = speaker_residual + speaker_hidden_states
            #     pass
            # else:
            speaker_hidden_states = speaker_residual + speaker_hidden_states

            # add cross-attn to positions 3,4 of present_key_value tuple
            speaker_present_key_value = speaker_present_key_value + speaker_cross_attn_present_key_value
        speaker_residual = speaker_hidden_states
        speaker_hidden_states = self.speaker_final_layer_norm(speaker_hidden_states)
        speaker_hidden_states = self.activation_fn(self.speaker_fc1(speaker_hidden_states))
        speaker_hidden_states = nn.functional.dropout(speaker_hidden_states, p=self.activation_dropout, training=self.training)
        speaker_hidden_states = self.speaker_fc2(speaker_hidden_states)
        speaker_hidden_states = nn.functional.dropout(speaker_hidden_states, p=self.dropout, training=self.training)
        speaker_hidden_states = speaker_residual + speaker_hidden_states
        #####################################################################################################################

        # outputs = (hidden_states,)
        spk_outputs = (speaker_hidden_states, )

        if output_attentions:
            # outputs += (self_attn_weights, cross_attn_weights)
            spk_outputs += (self_speaker_attn_weights, speaker_cross_attn_weights)

        if use_cache:
            # outputs += (present_key_value,)
            spk_outputs += (speaker_present_key_value, )

        return spk_outputs

class SpeakerEncoder(WhisperPreTrainedModel):
    def __init__(self, config: ConditionalSpeakerGenerationConfig):
        super().__init__(config)
        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop

        embed_dim = config.d_model
        self.num_mel_bins = config.num_mel_bins
        self.padding_idx = config.pad_token_id
        self.max_source_positions = config.max_source_positions
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0

        self.conv1 = nn.Conv1d(self.num_mel_bins, embed_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1)

        self.embed_positions = nn.Embedding(self.max_source_positions, embed_dim)
        self.embed_positions.requires_grad_(False)
        
        self.layers = nn.ModuleList([WhisperEncoderLayer(config) for _ in range(config.speaker_encoder_layers)])
        self.layer_norm = nn.LayerNorm(config.d_model)

    def _freeze_parameters(self):
        for param in self.parameters():
            param.requires_grad = False
        self._requires_grad = False

    def get_input_embeddings(self) -> nn.Module:
        return self.conv1

    def set_input_embeddings(self, value: nn.Module):
        self.conv1 = value

    def forward(
        self,
        input_features
    ):
        inputs_embeds = nn.functional.gelu(self.conv1(input_features))
        inputs_embeds = nn.functional.gelu(self.conv2(inputs_embeds))

        inputs_embeds = inputs_embeds.permute(0, 2, 1)
        embed_pos = self.embed_positions.weight

        hidden_states = inputs_embeds + embed_pos
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        
        for idx, encoder_layer in enumerate(self.layers):
            layer_outputs = encoder_layer(
                hidden_states,
                None,
                layer_head_mask=None,
                output_attentions=None,
            )
            hidden_states = layer_outputs[0]
        hidden_states = self.layer_norm(hidden_states)
        return hidden_states

class SpeakerDecoder(WhisperPreTrainedModel):
    main_input_name = "input_ids"

    def __init__(self, config: ConditionalSpeakerGenerationConfig):
        super().__init__(config)
        self.dropout = config.dropout
        self.layerdrop = config.decoder_layerdrop
        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_target_positions
        self.max_source_positions = config.max_source_positions
        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0
        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model, self.padding_idx)
        self.embed_positions = WhisperPositionalEmbedding(self.max_target_positions, config.d_model)
        self.layers = nn.ModuleList([SpeakerDecoderLayer(config, layer_idx=idx) for idx in range(config.speaker_decoder_layers)])
        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"
        self._use_sdpa = config._attn_implementation == "sdpa"
        self.spk_layer_norm = nn.LayerNorm(config.d_model)
        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        speaker_features=None,
        encoder_attention_mask=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                Indices can be obtained using [`WhisperTokenizer`]. See [`PreTrainedTokenizer.encode`] and
                [`PreTrainedTokenizer.__call__`] for details.

                [What are input IDs?](../glossary#input-ids)
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            encoder_hidden_states (`torch.FloatTensor` of shape `(batch_size, encoder_sequence_length, hidden_size)`, *optional*):
                Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
                of the decoder.
            head_mask (`torch.Tensor` of shape `(decoder_layers, decoder_attention_heads)`, *optional*):
                Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            cross_attn_head_mask (`torch.Tensor` of shape `(decoder_layers, decoder_attention_heads)`, *optional*):
                Mask to nullify selected heads of the attention modules in encoder to avoid performing cross-attention
                on hidden heads. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
                Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
                shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of
                shape `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

                Contains pre-computed hidden-states (key and values in the self-attention blocks and in the
                cross-attention blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

                If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those
                that don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of
                all `decoder_input_ids` of shape `(batch_size, sequence_length)`.
            inputs_embeds (`torch.FloatTensor` of
                shape `(batch_size, sequence_length, hidden_size)`, *optional*): Optionally, instead of passing
                `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more
                control over how to convert `input_ids` indices into associated vectors than the model's internal
                embedding lookup matrix.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if getattr(self.config, "_flash_attn_2_enabled", False):
            # 2d mask is passed through the layers
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        else:
            # 4d mask is passed through the layers
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask, input_shape, inputs_embeds, past_key_values_length
            )

        # embed positions
        if input_ids is not None:
            positions = self.embed_positions(input_ids, past_key_values_length=past_key_values_length)
        else:
            positions = self.embed_positions(inputs_embeds, past_key_values_length=past_key_values_length)

        hidden_states = inputs_embeds + positions
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        speaker_hidden_states = hidden_states
        
        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache = True` is incompatible with gradient checkpointing. Setting `use_cache = False`..."
                )
                use_cache = False
        # decoder layers
        # all_hidden_states = () if output_hidden_states else None
        # all_self_attns = () if output_attentions else None
        # all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None
        # next_decoder_cache = () if use_cache else None
        # spk decoder layers
        # encoder_speaker_features = self.speaker_transform(speaker_features)
        encoder_speaker_features = speaker_features
        
        spk_all_hidden_states = () if output_hidden_states else None
        spk_all_self_attns = () if output_attentions else None
        spk_all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None
        spk_next_decoder_cache = () if use_cache else None

        # check if head_mask/cross_attn_head_mask has a correct number of layers specified if desired
        for attn_mask, mask_name in zip([head_mask, cross_attn_head_mask], ["head_mask", "cross_attn_head_mask"]):
            if attn_mask is not None:
                assert attn_mask.size()[0] == (len(self.layers)), (
                    f"The `{mask_name}` should be specified for {len(self.layers)} layers, but it is for"
                    f" {head_mask.size()[0]}."
                )
        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if output_hidden_states:
                # all_hidden_states += (hidden_states,)
                spk_all_hidden_states += (speaker_features, )
            if self.training:
                dropout_probability = torch.rand([])
                if dropout_probability < self.layerdrop:
                    continue
            
            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    speaker_hidden_states,
                    attention_mask,
                    encoder_hidden_states,
                    encoder_speaker_features,
                    encoder_attention_mask,  # encoder attention mask
                    head_mask[idx] if head_mask is not None else None,
                    cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None,
                    None,  # past_key_value
                    output_attentions,
                    use_cache,
                )
            else:
                layer_outputs = decoder_layer(
                    # hidden_states,
                    speaker_hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_speaker_features=encoder_speaker_features,
                    encoder_attention_mask=encoder_attention_mask,
                    layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                    cross_attn_layer_head_mask=(
                        cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None
                    ),
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )
            speaker_hidden_states = layer_outputs[0]

            if use_cache:
                spk_next_decoder_cache += (layer_outputs[3 if output_attentions else 1],)

            if output_attentions:
                spk_all_self_attns += (layer_outputs[1],)

                if encoder_hidden_states is not None:
                    spk_all_cross_attentions += (layer_outputs[2],)

        speaker_hidden_states = self.spk_layer_norm(speaker_hidden_states)
        # add hidden states from the last decoder layer
        if output_hidden_states:
            spk_all_hidden_states += (speaker_hidden_states, )

        next_cache = spk_next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, spk_all_hidden_states, spk_all_self_attns, spk_all_cross_attentions]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=speaker_hidden_states,
            past_key_values=next_cache,
            hidden_states=spk_all_hidden_states,
            attentions=spk_all_self_attns,
            cross_attentions=spk_all_cross_attentions,
        )
    
def _compute_new_attention_mask(hidden_states: torch.Tensor, seq_lens: torch.Tensor):
    """
    Computes an attention mask of the form `(batch, seq_len)` with an attention for each element in the batch that
    stops at the corresponding element in `seq_lens`.

    Args:
        hidden_states (`torch.FloatTensor` of shape `(batch, seq_len, *)`):
            The sequences to mask, where `*` is any number of sequence-specific dimensions including none.
        seq_lens (`torch.Tensor` of shape `(batch)`:
            Each element represents the length of the sequence at the same index in `hidden_states`

    Returns:
        `torch.FloatTensor`: The float attention mask of shape `(batch, seq_len)`
    """
    batch_size, mask_seq_len = hidden_states.shape[:2]

    indices = torch.arange(mask_seq_len, device=seq_lens.device).expand(batch_size, -1)

    bool_mask = indices >= seq_lens.unsqueeze(1).expand(-1, mask_seq_len)

    mask = hidden_states.new_ones((batch_size, mask_seq_len))

    mask = mask.masked_fill(bool_mask, 0)

    return mask

class ConditionalSpeakerGeneration(PreTrainedModel):
    config_class = ConditionalSpeakerGenerationConfig
    def __init__(self, config: ConditionalSpeakerGenerationConfig):
        super().__init__(config)    
        self.speaker_encoder = SpeakerEncoder(config)
        self.speaker_decoder = SpeakerDecoder(config)
        self.spk_out = nn.Linear(config.d_model, self.config.spk_hidden_size)
        self.asr_model = None


    def get_asr_model(self):
        if self.asr_model is not None:
            return self.asr_model
        from transformers import WhisperForConditionalGeneration
        # from transformers import WhisperForConditionalGeneration, WhisperProcessor
        
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype

        model_name_or_path, cache_path = self.config.asr_model_name_or_path, self.config.asr_model_cache_dir

        asr_model = WhisperForConditionalGeneration.from_pretrained(model_name_or_path, cache_dir=cache_path).eval().to(device=device, dtype=dtype)
        # asr_model_processor = WhisperProcessor.from_pretrained(model_name_or_path, cache_dir=cache_path)
        
        self.asr_model = {
            "model": asr_model,
            # "processor": asr_model_processor
        }
        return self.asr_model

    def prepare_encoder_attention_mask(self, encoder_outputs:torch.FloatTensor, audio_lengths:torch.LongTensor, tgt_len: torch.LongTensor):
        sub_sampled_lengths = torch.tensor([min(1500, math.ceil(audio_length / 16000 / 0.02)) for audio_length in audio_lengths]).to(audio_lengths.device)
        encoder_attention_mask = _compute_new_attention_mask(
            hidden_states=encoder_outputs, seq_lens=sub_sampled_lengths
        )
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        encoder_attention_mask = _prepare_4d_attention_mask(
            encoder_attention_mask, tgt_len=tgt_len, dtype=encoder_outputs.dtype
        )
        return encoder_attention_mask

    def forward(
        self,
        input_features: Optional[torch.FloatTensor],
        acoustic_features: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        input_lengths: Optional[torch.LongTensor] = None,

        # For training
        labels: Optional[torch.LongTensor] = None,
        spk_labels: Optional[torch.LongTensor] = None,
        spk_embedding: Optional[torch.FloatTensor] = None,
    ):  
        speaker_features = self.speaker_encoder(input_features)

        if acoustic_features is None:
            with torch.no_grad():
                acoustic_features = self.get_asr_model()['model'].model.encoder(input_features).last_hidden_state

        encoder_attention_mask = None
        if input_lengths is not None:
            encoder_attention_mask = self.prepare_encoder_attention_mask(encoder_outputs=acoustic_features, 
                                                                         audio_lengths=input_lengths, 
                                                                         tgt_len=decoder_input_ids.size(1))
        speaker_embedding = self.speaker_decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=acoustic_features,
            speaker_features=speaker_features,
            encoder_attention_mask=encoder_attention_mask
        )
        word_speaker_embedding = self.spk_out(speaker_embedding.last_hidden_state)
        
        loss = None
        if spk_labels is not None:
            loss_spk = nn.CosineEmbeddingLoss()
            loss_pairwise = nn.MSELoss()

            spk_embedding = spk_embedding[0]            
            output_spk_embedding = word_speaker_embedding[spk_labels >= 0]
            target_spk_embedding = torch.stack([spk_embedding[spk_labels[selected_idx[0]][selected_idx[1]]] for selected_idx in torch.stack(torch.where(spk_labels >= 0)).T])
            
            # pairwise cosine similarity output spk embedding
            output_spk_embedding_norm = torch.nn.functional.normalize(output_spk_embedding, p=2, dim=1)    
            output_pairwise = torch.mm(output_spk_embedding_norm, output_spk_embedding_norm.T)
            
            # pairwise cosine similarity target spk embedding
            target_spk_embedding_norm = torch.nn.functional.normalize(target_spk_embedding, p=2, dim=1)
            target_pairwise = torch.mm(target_spk_embedding_norm, target_spk_embedding_norm.T)
            
            # pairwise cosine similarity output spk embedding and target spk embedding
            output_target_pairwise = torch.mm(output_spk_embedding_norm, target_spk_embedding_norm.T)
            
            # mse loss between output_pairwise and target_pairwise
            loss_output_target = loss_pairwise(output_pairwise, target_pairwise)
            
            # mse loss between output_target_pairwise and target_pairwise
            loss_output_target_target = loss_pairwise(output_target_pairwise, target_pairwise)
            
            posivite_spk_loss = loss_spk(output_spk_embedding, target_spk_embedding, torch.ones(output_spk_embedding.size(0), dtype=torch.long, device=spk_labels.device))

            loss = posivite_spk_loss + loss_output_target + loss_output_target_target
            
            if random.random() < 0.002:
                print(f"\nposivite_spk_loss: {posivite_spk_loss}")
                print(f"loss_output_vs_target: {loss_output_target}")
                print(f"loss_output_target_vs_target: {loss_output_target_target}")

            ################Check ASR pretrained mode####################
            # with torch.no_grad():
            #     asr_outputs = self.get_asr_model()['model'].model(
            #         input_features,
            #         decoder_input_ids=decoder_input_ids,
            #     )
            #     lm_logits = self.get_asr_model()['model'].proj_out(asr_outputs[0])

            #     asr_loss = None
            #     if labels is not None:
            #         loss_fct = nn.CrossEntropyLoss()
            #         # move labels to correct device to enable PP
            #         labels = labels.to(lm_logits.device)
            #         asr_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.reshape(-1))
            #     print("asr loss", asr_loss)
            #########################################


        return Seq2SeqLMOutput(
            loss=loss,
            logits=word_speaker_embedding
        )