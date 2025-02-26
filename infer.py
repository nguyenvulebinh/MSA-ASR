import sys
import os
sys.path.append('./')
os.environ["CUDA_VISIBLE_DEVICES"]="5"
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from src.modeling_sa import ConditionalSpeakerGeneration
from transformers.modeling_outputs import BaseModelOutput
import torch
import torchaudio
import time


def extract_word_speaker_embedding(spk_attribute_model, asr_model, processor, audios, acoustic_features=None, words_batch=None, language = 'en', use_attention_mask=True):
    audio_features = processor(audios, sampling_rate=16000, return_tensors='pt')
    device = next(spk_attribute_model.parameters()).device
    dtype = next(spk_attribute_model.parameters()).dtype
    
    audio_features = audio_features['input_features'].to(device, dtype)
    with torch.no_grad():
        if words_batch is not None:
            decoder_input_ids = processor.tokenizer.pad(processor.tokenizer.batch_encode_plus(
                [
                    f"<|startoftranscript|><|{language}|><|transcribe|><|notimestamps|> "+' '.join(words) + "<|endoftext|>" for words in words_batch
                ], 
                add_special_tokens=False
            ), return_tensors="pt")['input_ids'].to(device)
        else:
            decoder_input_ids = None
        
        if decoder_input_ids is None or acoustic_features is None:
            if decoder_input_ids is None:
                asr_model_output = asr_model.generate(
                    audio_features,
                    decoder_input_ids=processor.tokenizer.batch_encode_plus([f"<|startoftranscript|><|{language}|><|transcribe|><|notimestamps|>"] * len(audios), return_tensors="pt", add_special_tokens=False)['input_ids'].to(device),
                    return_dict_in_generate=True,
                    output_scores=True,
                    output_hidden_states=True,
                )
                acoustic_features = asr_model_output.encoder_hidden_states[-1]
                decoder_input_ids = asr_model_output.sequences
            
            if acoustic_features is None:
                acoustic_features = asr_model.model.encoder(audio_features).last_hidden_state

        if use_attention_mask:
            input_lengths = torch.tensor([len(audio) for audio in audios]).to(device)
        else:
            input_lengths = None

        spk_embedding = spk_attribute_model(
            input_features=audio_features,
            acoustic_features=acoustic_features, 
            decoder_input_ids=decoder_input_ids,
            input_lengths=input_lengths
        ).logits
    
    output_ids = decoder_input_ids[:, 1:]
    spk_embedding = spk_embedding[:, :-1]
    
    batch_output = []
    for idx in range(len(output_ids)):
        sample_words = []
        sample_word_spk_embedding = []
        current_word = []
        current_word_embedding = []
        for token_id, token_spk_embedding in zip(output_ids[idx], spk_embedding[idx]):
            token = processor.tokenizer.decode([token_id], skip_special_tokens=True)
            if len(token) == 0:
                continue
            if token.startswith(' '):
                if len(current_word) > 0:
                    sample_words.append(''.join(current_word).strip())
                    sample_word_spk_embedding.append(torch.stack(current_word_embedding).mean(dim=0).detach().cpu().numpy().tolist())
                    current_word = []
                    current_word_embedding = []
            current_word.append(token)
            current_word_embedding.append(token_spk_embedding)
        if len(current_word) > 0:
            sample_words.append(''.join(current_word).strip())
            sample_word_spk_embedding.append(torch.stack(current_word_embedding).mean(dim=0).detach().cpu().numpy().tolist())
                    
        batch_output.append([sample_words, sample_word_spk_embedding])
    
    ## Check if the output is correct
    if words_batch is not None:
        for i in range(len(batch_output)):
            try:
                assert len(batch_output[i]) == len(words_batch[i]), f"{len(batch_output[i])} != {len(words_batch[i])}"
                assert [w[0] for w in batch_output[i]] == words_batch[i], f"{[w[0] for w in batch_output[i]]} != {words_batch[i]}"
            except:
                pass    

    return batch_output

def add_prefix_tokens(processor, prefix, forced_decoder_ids):
    if len(prefix) > 0:
        prompt_ids = processor.get_prompt_ids(prefix).tolist()[1:]
        for wid in prompt_ids:
            forced_decoder_ids.append((len(forced_decoder_ids) + 1, wid))

def infer_batch(audio_wavs, asr_model, asr_processor, spk_attribute_model, prefix="", transcripts=[], input_language="en", task="transcribe", audio_sample_rate=16000, beam_size=4):
    # get device based on the model parameters
    device = next(asr_model.parameters()).device
    dtype = next(asr_model.parameters()).dtype
    print("prefix:",prefix)
    possible_languages = None if input_language == "None" else input_language.split("+")
    print("Possible languages:", possible_languages)
    input_values = torch.cat([asr_processor(item, sampling_rate=audio_sample_rate, return_tensors="pt").input_features for item in audio_wavs], dim=0).to(device, dtype)
    print("Input transcripts:",transcripts)
    
    if input_language != "None" and not "+" in input_language and None not in transcripts:
        words_batch = [t.split() for t in transcripts]
        start_time = time.time()
        saasr_output = extract_word_speaker_embedding(
            spk_attribute_model=spk_attribute_model,
            asr_model=asr_model,
            processor=asr_processor,
            audios=[item.numpy() for item in audio_wavs],
            acoustic_features=None,
            words_batch=words_batch,
        )
        print("SAASR time: {:.2f}s".format(time.time()-start_time))
        return transcripts, [input_language] * len(transcripts), saasr_output
    
    
    if input_language != "None" and not "+" in input_language:
        forced_decoder_ids = asr_processor.get_decoder_prompt_ids(language=input_language, task=task)
        add_prefix_tokens(asr_processor, prefix, forced_decoder_ids)
    else:
        forced_decoder_ids = asr_processor.get_decoder_prompt_ids(language="en", task=task)[1:]

        output = asr_model.generate(
            input_values, 
            max_new_tokens=1,
            forced_decoder_ids=forced_decoder_ids,
            return_dict_in_generate=True,
            output_scores=True,
            output_hidden_states=True,
        ) # Predicts the language
        
        predicted_ids = output.sequences[:,1]

        forced_decoder_ids = asr_processor.get_decoder_prompt_ids(language="en", task=task)
        add_prefix_tokens(asr_processor, prefix, forced_decoder_ids)
        pred_to_indices = {}
        for i,pred in enumerate(predicted_ids.tolist()):
            if pred not in pred_to_indices:
                pred_to_indices[pred] = [i]
            else:
                pred_to_indices[pred].append(i)

        outputs = {}
        lids = {}
        saasr = {}
        for pred, indices in pred_to_indices.items():
            forced_decoder_ids[0] = (forced_decoder_ids[0][0],pred)
            encoder_outputs = BaseModelOutput(last_hidden_state=output["encoder_hidden_states"][-1][indices])

            start_time = time.time()
            asr_model_output = asr_model.generate(
                input_values[indices], 
                forced_decoder_ids=forced_decoder_ids,
                no_repeat_ngram_size=6,
                encoder_outputs=encoder_outputs,
                return_dict_in_generate=True,
                output_scores=True,
                output_hidden_states=True,
                num_beams=beam_size,
            )
            print("ASR time: {:.2f}s".format(time.time()-start_time))
            acoustic_features = output["encoder_hidden_states"][-1][indices]
            predicted_ids2 = asr_model_output.sequences
            text_output_raw = asr_processor.batch_decode(predicted_ids2, skip_special_tokens=True)
            
            # SAASR inference
            words_batch = [t.split() for t in text_output_raw]
            print(predicted_ids2[:,1])
            language = asr_processor.batch_decode(predicted_ids2[:,1])[0].strip("<|>")
            
            if (possible_languages is not None) and (language not in possible_languages):
                print("Language not in possible languages:", language, "Force to output prefix")
                text_output_raw = [prefix for i in range(len(indices))]
                words_batch = [prefix.split() for i in range(len(indices))]

            start_time = time.time()
            saasr_output = extract_word_speaker_embedding(
                spk_attribute_model=spk_attribute_model,
                asr_model=asr_model,
                processor=asr_processor,
                audios=[item.numpy() for item in [audio_wavs[i_a] for i_a in indices]],
                acoustic_features=acoustic_features,
                words_batch=words_batch,
                language=language
            )
            print("SAASR time: {:.2f}s".format(time.time()-start_time))
            print("Output {}:".format(language), text_output_raw)
            
            for o,i,lid,spk_embed in zip(text_output_raw,indices,asr_processor.batch_decode(predicted_ids2[:,1]), saasr_output):
                outputs[i] = o
                lids[i] = lid
                saasr[i] = spk_embed

        return [outputs[i] for i in range(len(outputs))], [lids[i][2:-2] for i in range(len(outputs))], [saasr[i] for i in range(len(outputs))]
    
    with torch.no_grad():
        start_time = time.time()
        asr_model_output = asr_model.generate(
            input_values, 
            forced_decoder_ids=forced_decoder_ids,
            no_repeat_ngram_size=6,
            return_dict_in_generate=True,
            output_scores=True,
            output_hidden_states=True,
            num_beams=beam_size,
        )
        print("ASR time: {:.2f}s".format(time.time()-start_time))
    
    acoustic_features = asr_model_output.encoder_hidden_states[-1]
    predicted_ids = asr_model_output.sequences
    text_output_raw = asr_processor.batch_decode(predicted_ids, skip_special_tokens=True)
    lids = asr_processor.batch_decode(predicted_ids[:,1])
    language = lids[0].strip("<|>")
    
    if (possible_languages is not None) and (language not in possible_languages):
        print("Language not in possible languages:", language, "Force to output prefix")
        text_output_raw = [prefix for i in range(len(indices))]
        words_batch = [prefix.split() for i in range(len(indices))]
    
    # SAASR inference
    words_batch = [t.split() for t in text_output_raw]
    start_time = time.time()
    saasr_output = extract_word_speaker_embedding(
        spk_attribute_model=spk_attribute_model,
        asr_model=asr_model,
        processor=asr_processor,
        audios=[item.numpy() for item in audio_wavs],
        acoustic_features=acoustic_features,
        words_batch=words_batch,
        language=language
    )
    print("SAASR time: {:.2f}s".format(time.time()-start_time))
    print("Output:",text_output_raw)
    
    return text_output_raw, [lid[2:-2] for lid in lids], saasr_output


if __name__ == "__main__":
    
    cache_dir = "./cache"
    
    asr_model_name = 'openai/whisper-large-v2'
    asr_model = WhisperForConditionalGeneration.from_pretrained(asr_model_name, cache_dir=cache_dir).eval()
    asr_processor = WhisperProcessor.from_pretrained(asr_model_name, cache_dir=cache_dir)
        
    spk_attribute_model_name = 'nguyenvulebinh/MSA-ASR'
    spk_attribute_model = ConditionalSpeakerGeneration.from_pretrained(spk_attribute_model_name, cache_dir=cache_dir).eval()
    
    if torch.cuda.is_available():
        spk_attribute_model = spk_attribute_model.cuda().half()
        asr_model = asr_model.cuda().half()
        

    # audio_file = "./resource/sample_vi.wav"
    audio_file = "./resource/sample_en.wav"
    
    
    _, _, output = infer_batch(
        [torchaudio.load(audio_file)[0][0]], 
        asr_model, 
        asr_processor, 
        spk_attribute_model,
        
        # transcripts=[None],
        # transcripts=["""Chuột là loài động vật thuộc bộ gặm nhấm bộ chuột, khối lượng cơ thể chúng khoảng 10-25 gram. Gà là loài ăn tạp, thường bởi đất hạt tìm cây, côn trùng, thằn lằn hoặc chuột nhắt con, lông nhọn trên cổ và lưng thường có màu sáng đậm màu hơn."""],
        # input_language="vi",
        
        # transcripts=[None],
        transcripts=["Okay, let's just talk a bit. What have you been working on? I am working on the ASR combined with the speaker diarization."],
        input_language="en",
    )
    
    # print(output_text)
    output_words = output[0][0]
    output_embedding = output[0][1]
    
    # Simple spk change detection
    scores = torch.nn.functional.cosine_similarity(
        torch.tensor(output_embedding)[:-1],
        torch.tensor(output_embedding)[1:]
    )
    threshold = 0.8
    # print(scores)
    spk_change = (scores < threshold).numpy().tolist() + [False]
    words_with_spk_change = ' '.join(['<sc> ' + w if c else w for w, c in zip(output_words, spk_change)])
    print(words_with_spk_change)
    
    