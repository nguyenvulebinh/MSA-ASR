# MSA-ASR
Multilingual Speaker-Attributed Automatic Speech Recognition

### Demo

<video src="https://huggingface.co/nguyenvulebinh/MSA-ASR/resolve/main/demo_sa-asr.mp4" width="640" height="480" controls></video>

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

*Example*

Audio

<audio controls>
  <source src="https://huggingface.co/nguyenvulebinh/MSA-ASR/resolve/main/sample_augment.wav" type="audio/wav">
  Your browser does not support the audio element.
</audio>


Label:

```code
spk_1 A 0.00 1.58 »spk_1
spk_1 A 0.00 1.58 Pacifica
spk_1 A 1.58 0.68 continues
spk_1 A 2.27 0.52 today
spk_1 A 2.79 0.24 to
spk_1 A 3.03 0.20 be
spk_1 A 3.23 0.14 a
spk_1 A 3.37 0.54 listener
spk_1 A 3.91 0.80 supported
spk_1 A 4.71 0.70 network
spk_1 A 5.42 0.38 of
spk_2 A 5.80 0.12 »spk_2
spk_2 A 5.80 0.12 At
spk_2 A 5.92 0.42 home,
spk_2 A 6.34 0.18 an
spk_2 A 6.52 0.38 Aed
spk_2 A 6.90 0.26 is
spk_2 A 7.16 0.18 an
spk_2 A 7.34 0.56 automated
spk_2 A 7.90 0.60 external
spk_2 A 8.50 0.90 defibrillator.
spk_2 A 9.40 0.40 It's
spk_2 A 9.81 0.08 the
spk_2 A 9.89 0.36 device
spk_2 A 10.25 0.08 you
spk_2 A 10.33 0.16 use
spk_2 A 10.49 0.12 when
spk_2 A 10.61 0.10 your
spk_2 A 10.73 0.16 heart
spk_2 A 10.89 0.18 goes
spk_2 A 11.07 0.12 into
spk_2 A 11.19 0.38 cardiac
spk_2 A 11.57 0.38 arrest
spk_2 A 11.95 0.18 to
spk_2 A 12.13 0.36 shock
spk_2 A 12.49 0.14 it
spk_2 A 12.63 0.28 back
spk_2 A 12.91 0.22 into
spk_2 A 13.13 0.06 a
spk_2 A 13.19 0.32 normal
spk_2 A 13.51 0.88 rhythm.
spk_1 A 14.40 1.38 »spk_1
spk_1 A 14.40 1.38 stations.
```

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
