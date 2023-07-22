# Real-Time Voice Cloning v2

### What is this?
It is an improved version of [Real-Time-Voice-Cloning](https://github.com/CorentinJ/Real-Time-Voice-Cloning). Our emotion voice cloning implementation is [here](https://github.com/liuhaozhe6788/voice-cloning-collab/tree/add_emotion)!

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
3. [VCTK](https://datashare.ed.ac.uk/handle/10283/3443): used for training and validation

**Synthesizer preprocessing:** 
```
python synthesizer_preprocess_audio.py <datasets_root>
python synthesizer_preprocess_embeds.py <datasets_root>/SV2TTS/synthesizer
```

**Synthesizer training:** 
```
python synthesizer_train.py <model_id> <datasets_root>/SV2TTS/synthesizer --use_tb
```
if you want to monitor the training progress, run
```
tensorboard --logdir log/synthesizer --host localhost --port 8088
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
python vocoder_train.py <model_id> <datasets_root> --use_tb
```
if you want to monitor the training progress, run
```
tensorboard --logdir log/vocoder --host localhost --port 8080
```
**Note:**

Training breakpoints are saved periodically, so you can run the training command and resume training when the breakpoint exists.

## Inference 

**Terminal:** 
```
python demo_cli.py
```
First input the number of audios, then input the audio file paths, then input the text message. The attention alignments and mel spectrogram are stored in syn_results/. The generated audio is stored in out_audios/.

**GUI demo:**
```
python demo_toolbox.py
```
## Dimension reduction visualization
**Download dataset:** 

[LibriSpeech](https://www.openslr.org/12): test-other
(extract as <datasets_root>/LibriSpeech/<dataset_name>)

**Preprocessing:** 
```
python encoder_test_preprocess.py <datasets_root>
```

**Visualization:**
```
python encoder_test_visualization.py <model_id> <datasets_root>
```
The results are saved in dim_reduction_results/.

## Pretrained models
You can download the pretrained model from [this](https://drive.google.com/drive/folders/19fhjjAbWq60zv1Bl6Y51snGbG1r5kaN2) and extract as saved_models/20230609

## Demo results
<div align = "center">
<table style="width:100%">
  <thead>
    <tr>
      <th width="550" > Input Text</th>
      <th>Synthetic Audio</th>
    </tr>
  </thead>
  <tbody>
   <tr>
      <td>Life was like a box of chocolates, you never know what you're gonna get.</td>
      <td align = "center">
      <audio controls><source src="samples/1320_00000.mp3"></audio> 
      </td>
    </tr>
    <tr>
      <td>早上好，今天是2020/10/29，最低温度是-3°C。</td>
      <td align = "center">
      <a href="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/parakeet_espnet_fs2_pwg_demo/tn_g2p/parakeet/001.wav" rel="nofollow">
            <img align="center" src="./docs/images/audio_icon.png" width="200" style="max-width: 100%;"></a><br>
      </td>
    </tr>
    <tr>
      <td>季姬寂，集鸡，鸡即棘鸡。棘鸡饥叽，季姬及箕稷济鸡。鸡既济，跻姬笈，季姬忌，急咭鸡，鸡急，继圾几，季姬急，即籍箕击鸡，箕疾击几伎，伎即齑，鸡叽集几基，季姬急极屐击鸡，鸡既殛，季姬激，即记《季姬击鸡记》。</td>
      <td align = "center">
      <a href="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/jijiji.wav" rel="nofollow">
            <img align="center" src="./docs/images/audio_icon.png" width="200" style="max-width: 100%;"></a><br>
      </td>
    </tr>
    <tr>
      <td>大家好，我是 parrot 虚拟老师，我们来读一首诗，我与春风皆过客，I and the spring breeze are passing by，你携秋水揽星河，you take the autumn water to take the galaxy。</td>
      <td align = "center">
      <a href="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/labixiaoxin.wav" rel="nofollow">
            <img align="center" src="./docs/images/audio_icon.png" width="200" style="max-width: 100%;"></a><br>
      </td>
    </tr>
    <tr>
      <td>宜家唔系事必要你讲，但系你所讲嘅说话将会变成呈堂证供。</td>
      <td align = "center">
      <a href="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/chengtangzhenggong.wav" rel="nofollow">
            <img align="center" src="./docs/images/audio_icon.png" width="200" style="max-width: 100%;"></a><br>
      </td>
    </tr>
    <tr>
      <td>各个国家有各个国家嘅国歌</td>
      <td align = "center">
      <a href="https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/gegege.wav" rel="nofollow">
            <img align="center" src="./docs/images/audio_icon.png" width="200" style="max-width: 100%;"></a><br>
      </td>
    </tr>
  </tbody>
</table>

</div>
