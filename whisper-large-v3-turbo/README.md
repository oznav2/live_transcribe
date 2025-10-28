---
library_name: transformers
license: apache-2.0
datasets:
- ivrit-ai/crowd-transcribe-v5
- ivrit-ai/crowd-recital-whisper-training
- ivrit-ai/knesset-plenums-whisper-training
language:
- he
metrics:
- wer
base_model:
- openai/whisper-large-v3-turbo
pipeline_tag: automatic-speech-recognition
---

# Model Card for Model ID

This model is a Hebrew finetune (continued training) of the OpenAI Whisper Large v3 Turbo model.


## Model Details

### Model Description

- **Developed by:** ivrit-ai
- **Language(s) (NLP):** Hebrew
- **License:** Apache-2.0
- **Finetuned from model** openai/whisper-large-v3-turbo
- **Training Date** Apr 2025

## Bias, Risks, and Limitations

Language detection capability of this model has been degraded during training - it is intended for mostly-hebrew audio transcription.
Language token should be explicitly set to Hebrew.

Additionally, the tanslation task was not trained and also degraded. This model would not be able to translate in any reasonable capacity.

## How to Get Started with the Model

Please follow the original [model card](https://huggingface.co/openai/whisper-large-v3-turbo#usage) for usage details - replacing with this model name.
You can also fine other weight formats ad quantizations on the [ivrit ai](https://huggingface.co/ivrit-ai) HF page.

We created some simple example scripts using this model and weights for other inference runtimes.
Find those in the ["examples"](https://github.com/ivrit-ai/asr-training/tree/master/examples) folder within the training GitHub repo.

## Training Details

### Training Data

This model was trained on the following datasets:

- [ivrit-ai/crowd-transcribe-v5](https://huggingface.co/datasets/ivrit-ai/crowd-transcribe-v5) - Publicly accessible audio sources have been crowd-transcribed segment-by-segment - ~300h
- [ivrit-ai/crowd-recital-whisper-training](https://huggingface.co/datasets/ivrit-ai/crowd-recital-whisper-training) - Crowd-sourced recording of Wikipedia article snippets. ~50h
- [ivrit-ai/knesset-plenums-whisper-training](https://huggingface.co/datasets/ivrit-ai/knesset-plenums-whisper-training) - A subset of a Knesset (Israeli house of representatives) plenum protocols. ~4700h

### Training Procedure

This model was trained in two main phases:
- Knesset based pre-training - over all ~4700h of data - 3 epochs, ~48h run
- Mixed post-training over all crowd-transcribe-v5 (300h), crowd-recital-whisper-training (50h) and highest-quality filtered knessets data (150h) - 2 epochs
 - Interleaving of datasets with sampling probs: (0.9, 0.025, 0.075) respectively
 - Note that crowd-transcribe-v5 has about 5x shorter samples on average thus the over-sampling.

This model is a weighted-average of the 2 lowest eval loss checkpoints (From around the end of epoch 2) from two seprate runs with the same setup.
Training code can be found on the ivrit-ai Github [here](https://github.com/ivrit-ai/asr-training)

#### Preprocessing

The "Crowd Recital" and "Knesset" datasets contain timestamps and previous text following the Whisper expected inputs.
Timestamps were used from 40% of samples from those datasets, and 50% of the previous text was used.

The "Crowd Transcribe" datasets has no timestamps or previous text and this preprocessing only included melspec feature extraction and text encoding.

Preprocessing code can be found within the training code [repository](https://github.com/ivrit-ai/asr-training).

Datasets were interleaved with 0.15:0.8:0.05 ratio (knesset:crowd-transcribe:crowd-recital).

#### Training Hyperparameters

- **Training regime:** bf16 mixed precision with sdpa
- **Learning Rate:** 1e-5, Linear decay, 800 steps warmup for 3 epochs
- **Batch Size:** 32

#### Training Hardware / Duration

- **GPU Type:** 8 x Nvidia A40 machine
- **Duration:** ~55h run across both phases

## Evaluation

Please refer to the [ivrit-ai/hebrew-transcription-leaderboard](https://huggingface.co/spaces/ivrit-ai/hebrew-transcription-leaderboard)