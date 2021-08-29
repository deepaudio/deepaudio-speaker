## Content
- [**What is deepaudio-speaker?**](https://github.com/deepaudio/deepaudio-speaker#what-is-deepaudio)
- [**Installation**](https://github.com/deepaudio/deepaudio-speaker#installation)
- [**Get Started**](https://github.com/deepaudio/deepaudio-speaker#get-started)
- [**Model Architecture**](https://github.com/deepaudio/deepaudio-speaker#model-architectures)
- [**How to contribute to deepaudio-speaker?**](https://github.com/deepaudio/deepaudio-speaker#How-to-contribute-to-deepaudio-speaker)
- [**Acknowledge**](https://github.com/deepaudio/deepaudio-speaker#Acknowledge)

## What is deepaudio-speaker?

Deepaudio-speaker is a framework for training neural network based speaker embedders. It supports online audio augmentation thanks to torch-audiomentation. It inlcudes or will include  popular neural network architectures and losses used for speaker embedder. 

To make it easy to use various functions such as mixed-precision, multi-node training, and TPU training etc, I introduced PyTorch-Lighting and Hydra in this framework (just like what [pyannote-audio](https://github.com/pyannote/pyannote-audio) and [openspeech](https://github.com/openspeech-team/openspeech) do).    

Deepaudio-tts is coming soon.

## Installation
```
conda create -n deepaudio python=3.8.5
conda activate deepaudio
conda install numpy cffi
conda install libsndfile=1.0.28 -c conda-forge
git clone https://github.com/deepaudio/deepaudio-speaker.git
cd deepaudio-speaker
pip install -e .
```

## Get Started

### Supported Datasets

####Voxceleb2
* [Download VoxCeleb dataset](http://www.robots.ox.ac.uk/~vgg/data/voxceleb/) and follow [this script](https://github.com/pyannote/pyannote-db-voxceleb/issues/10#issuecomment-702638328) to obtain this kind of directory structure:

```
/path/to/voxceleb/voxceleb1/dev/wav/id10001/1zcIwhmdeo4/00001.wav
/path/to/voxceleb/voxceleb1/test/wav/id10270/5r0dWxy17C8/00001.wav
/path/to/voxceleb/voxceleb2/dev/aac/id00012/21Uxsk56VDQ/00001.m4a
/path/to/voxceleb/voxceleb2/test/aac/id00017/01dfn2spqyE/00001.m4a
```

### Training examples
 - Example1: Train the `ecapa-tdnn` model with `fbank` features on GPU.
  
```
$ deepaudio-speaker-train  \
    dataset=voxceleb2 \
    dataset.dataset_path=/your/path/to/voxceleb2/dev/wav/ \
    model=clovaai_ecapa \
    model.channels=1024 \
    feature=fbank \
    lr_scheduler=reduce_lr_on_plateau \
    trainer=gpu \
    criterion=pyannote_aamsoftmax
```
- Example2: Extract speaker embedding with trained model.

Todo

## Model Architecture
[**ECAPA-TDNN**](https://arxiv.org/pdf/2005.07143.pdf) This is an unofficial implementation from @lawlict. Please find more details in this [link](https://github.com/lawlict/ECAPA-TDNN).

[**ECAPA-TDNN**](https://arxiv.org/pdf/2005.07143.pdf) This is implemented by @joonson. Please find more details in this [link](https://github.com/clovaai/voxceleb_trainer/issues/86#issuecomment-739991154).

[**ResNetSE34L**](https://arxiv.org/pdf/2003.11982.pdf) This is borrowed from [voxceleb trainer](https://github.com/clovaai/voxceleb_trainer).

[**ResNetSE34V2**](https://arxiv.org/pdf/2003.11982.pdf) This is borrowed from [voxceleb trainer](https://github.com/clovaai/voxceleb_trainer).

[**Resnet101**](https://arxiv.org/abs/2012.14952) This is proposed by BUT for speaker diarization. Please note that the feature used in this framework is different from [VB-HMM](https://github.com/BUTSpeechFIT/VBx) 

## How to contribute to deepaudio-speaker

It is a personal project. So I don't have enough gpu resources to do a lot of experiments. I appreciate any kind of feedback or contributions. Please feel free to make a pull requsest for some small issues like bug fixes, experiment results. If you have any questions, please [open an issue](https://github.com/deepaudio/deepaudio-speaker/issues).

## Acknowledge
I borrow a lot of codes from [openspeech](https://github.com/openspeech-team/openspeech) and [pyannote-audio](https://github.com/pyannote/pyannote-audio)