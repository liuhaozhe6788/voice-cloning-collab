# Real-Time Voice Cloning v2

### What is this?
It is an improved version of [Real-Time-Voice-Cloning](https://github.com/CorentinJ/Real-Time-Voice-Cloning).

## Installation
1. Install [ffmpeg](https://ffmpeg.org/download.html#get-packages). This is necessary for reading audio files.

2. Create a new conda environment with 
```
conda create -n rtvc python=3.7.13
```
3. Install [PyTorch](https://download.pytorch.org/whl/torch_stable.html).  Pick the proposed CUDA version if you have a GPU, otherwise pick CPU.
My torch version: `torch=1.9.1+cu111`
`torchvision=0.10.1+cu111`

4. Install the remaining requirements with 
```
pip install -r requirements.txt
```

5. Install spaCy model en_core_web_sm by 
`python -m spacy download en_core_web_sm`


## Training

### Encoder 

**Download dataset：** 

1. [LibriSpeech](https://www.openslr.org/12): train-other-500 for training, dev-other for validation
(extract as <datasets_root>/LibriSpeech/<dataset_name>)

2. [VoxCeleb1](https://mm.kaist.ac.kr/datasets/voxceleb/): Dev A - D for training, Test for validation as well as the metadata file `vox1_meta.csv` (extract as <datasets_root>/VoxCeleb1/ and <datasets_root>/VoxCeleb1/vox1_meta.csv)

3. [VoxCeleb2](https://mm.kaist.ac.kr/datasets/voxceleb/): Dev A - H for training, Test for validation
(extract as <datasets_root>/VoxCeleb2/)

**Encoder preprocessing：** 
```
python encoder_preprocess.py <datasets_root>
```

**Encoder training：** 

it is recommended to start visdom server for monitor training with
```
visdom
```
then start training with
```
python encoder_train.py <model_id> <datasets_root>/SV2TTS/encoder
```
### Synthesizer

**Download dataset：** 
1. [LibriSpeech](https://www.openslr.org/12): train-clean-100 and train-clean-360 for training, dev-clean for validation (extract as <datasets_root>/LibriSpeech/<dataset_name>)
2. [LibriSpeech alignments](https://drive.google.com/file/d/1WYfgr31T-PPwMcxuAq09XZfHQO5Mw8fE/view?usp=sharing): merge the directory structure with the LibriSpeech datasets you have downloaded (do not take the alignments from the datasets you haven't downloaded else the scripts will think you have them)

**Synthesizer preprocessing:** 
```
python synthesizer_preprocess_audio.py <datasets_root>
python synthesizer_preprocess_embeds.py <datasets_root>/SV2TTS/synthesizer
```

**Synthesizer training:** 
```
python synthesizer_train.py <model_id> <datasets_root>/SV2TTS/synthesizer
```
if you want to monitor the training progress, run
```
python update_plot.py syn
```
### Vocoder

**Download dataset：** 

The same as synthesizer. You can skip this if you already download synthesizer training dataset.

**Vocoder preprocessing:** 
```
python vocoder_preprocess.py <datasets_root>
```

**Vocoder training:** 
```
python vocoder_train.py <model_id> <datasets_root>
```
if you want to monitor the training progress, run
```
python update_plot.py voc
```
**Note:**

Training breakpoints are saved periodically, so you can run the training command and resume training when the breakpoint exists.

## Inference 

**Terminal:** 
```
python demo_cli.py
```
First input the number of audios, then input the audio file paths, then input the text message. The attention alignments and mel spectrogram are stored in syn_results/. The generated audio is stored in out_audios/.
## Dimension reduction visualization
**Download dataset：** 

[LibriSpeech](https://www.openslr.org/12): test-other
(extract as <datasets_root>/LibriSpeech/<dataset_name>)

**Preprocessing:** 
```
python encoder_test_preprocess.py <datasets_root>
```

**Visualization**
```
python encoder_test_visualization.py <model_id> <datasets_root>
```
The results are saved in dim_reduction_results/.

## Pretrained models
You can download the pretrained model from [this](https://drive.google.com/drive/folders/1oi5the9QxNbpOol_I5Qpr42hvoFXdhEF) and extract as saved_models/default
## What's new
**2022.05.19：** We calculated GE2E loss in encoder with CUDA rather than originally-configured CPU. It speeds up the encoder training speed.<br>
**2022.07.15：** We added Loss animation plot for synthesizer and vocoder.<br>
**2022.07.19：** We added response time and Griffin-Lim vocoder results for demo_toolbox.<br>
**2022.07.29：** We added model validation for encoder, synthesizer and vocoder.<br>
**2022.08.02：** We added voxceleb train and dev data for encoder. We added noise reduce method for the output wav from vocoder.
[noisereduce reference](https://github.com/timsainb/noisereduce)<br>
**2022.08.06：** We split the long text into short sentences using spacy for input of synthesizer. Make sure to install spaCy model en_core_web_sm by 
`python -m spacy download en_core_web_sm`<br>
**2022.09.02：** We set prop_decrease=0.6 for male and 0.9 for female in noisereduce function.(输出滤波，男女声使用不同的滤波参数)<br>
**2022.09.26：** We added speed adjustment(声音变速) for output audios using praat, install parselmouth using pip: `pip install praat-parselmouth`<br>
**2022.10.10：** We added voice filter functioning(声音美颜) for input audios, the weight ratio of the input audio embed and the standard audio embed is 7: 3. <br>
**2022.10.25：** We set small values(<0.06) to zeros in embed.(对嵌入向量较小值置零)<br>
**2022.10.26：** The split frequency for input audio is 170Hz. The split frequency for output noise reduce is 165Hz.<br>
**2022.12.01：** merge the single sentences to input.<br>
**2022.12.31：** added speaker embeddings dimension reduction visualzation results.<br>
**2023.01.01：** did more text preprocessing and text cleaning for TTS text input.<br>
**2023.02.17：** removed '#' char and convert '@' to "at".<br>