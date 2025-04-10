# [MSA-ASR](https://huggingface.co/nguyenvulebinh/MSA-ASR)
Multilingual Speaker-Attributed Automatic Speech Recognition

### Introduction

This repository provides an implementation of a Speaker-Attributed Automatic Speech Recognition model. The model performs both multilingual speech recognition and speaker embedding extraction, enabling speaker differentiation.

Model architecture

![MSA-ASR Model](https://github.com/nguyenvulebinh/MSA-ASR/blob/main/resource/model.png?raw=true)


### Setup

```
git clone git@github.com:nguyenvulebinh/MSA-ASR.git
cd MSA-ASR
conda create -n MSA-ASR python=3.10
conda activate MSA-ASR
pip install -r requirements.txt
```

Test script:

```
python infer.py
```

### Training Dataset

*From ASR to SA-ASR dataset:*

- Segment ASR data into single-speaker turns.
- Match turns into group which may come from the same speaker by using speaker embedding cosine similarity.
- Pick a few groups, each group a few turns.
- Concatenate turns in random order.

![MSA-ASR Dataset](https://github.com/nguyenvulebinh/MSA-ASR/blob/main/resource/sa_asr_data_pipeline.png?raw=true)

*In total:*

- 15.5M turns
- 14k audio hours
- English only

Dataset is openly available in [HF Dataset](https://huggingface.co/datasets/nguyenvulebinh/spk-attribute)

### Citation

```bibtex
@INPROCEEDINGS{10889116,
  author={Nguyen, Thai-Binh and Waibel, Alexander},
  booktitle={ICASSP 2025 - 2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={MSA-ASR: Efficient Multilingual Speaker Attribution with frozen ASR Models}, 
  year={2025},
  volume={},
  number={},
  pages={1-5},
  keywords={Training;Adaptation models;Limiting;Predictive models;Data models;Robustness;Multilingual;Data mining;Speech processing;Standards;speaker-attributed;asr;multilingual},
  doi={10.1109/ICASSP49660.2025.10889116}}

@INPROCEEDINGS{10446589,
  author={Nguyen, Thai-Binh and Waibel, Alexander},
  booktitle={ICASSP 2024 - 2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={Synthetic Conversations Improve Multi-Talker ASR}, 
  year={2024},
  volume={},
  number={},
  pages={10461-10465},
  keywords={Systematics;Error analysis;Knowledge based systems;Oral communication;Signal processing;Data models;Acoustics;multi-talker;asr;synthetic conversation},
  doi={10.1109/ICASSP48485.2024.10446589}}


```

### License

CC-BY-NC 4.0

### Contact

Contributions are welcome; feel free to create a PR or email me:

```
[Binh Nguyen](nguyenvulebinh[at]gmail.com)
```
