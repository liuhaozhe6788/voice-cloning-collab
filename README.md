# Voice-Cloning
## Installation
```
pip install -r requirements.txt
```
## Training Commands
**Encoder Preprocessing：** 
```
python encoder_preprocess.py <datasets_root>
```

**Encoder Training：** 
```
python encoder_train.py my_run <datasets_root>/SV2TTS/encoder
```

**Synthesizer Preprocessing:** 
```
python synthesizer_preprocess_audio.py <datasets_root>
python synthesizer_preprocess_embeds.py <datasets_root>/SV2TTS/synthesizer
```

**Synthesizer Training:** 
```
python synthesizer_train.py my_run <datasets_root>/SV2TTS/synthesizer
```

**Vocoder Preprocessing:** 
```
python vocoder_preprocess.py <datasets_root>
```

**Vocoder Training:** 
```
python vocoder_train.py my_run <datasets_root>
```


## Version updates
**2022.05.19：** We calculated GE2E loss in encoder with cuda rather than originally-configured CPU. It speeds up the encoder training speed.<br>
**2022.07.15：** We added Loss animation plot for synthesizer and vocoder.<br>
**2022.07.19：** We added response time and Griffin-Lim vocoder results for demo_toolbox.<br>
**2022.07.29：** We added model validation for encoder, synthesizer and vocoder.<br>
**2022.08.02：** We added voxceleb train and dev data for encoder. We added noise reduce method for the output wav from vocoder.<br>
noisereduce reference: https://github.com/timsainb/noisereduce<br>
**2022.08.06：** We split the long text into short sentences using spacy for input of synthesizer. Make sure to install English dataset en_core_web_sm, say by ***python -m spacy download en_core_web_sm***<br>
**2022.09.02：** We set prop_decrease=0.6 for male and 0.9 for female in noisereduce function.(输出滤波，男女声使用不同的滤波参数)<br>
**2022.09.26：** We added speed adjustment(声音变速) for output audios using praat, install parselmouth using pip: ***pip install praat-parselmouth***<br>
**2022.10.10：** We added voice filter functioning(声音美颜) for input audios, the weight ratio of the input audio embed and the standard audio embed is 7: 3. <br>
**2022.10.25：** We set small values(<0.06) to zeros in embed.(对嵌入向量较小值置零)<br>
**2022.10.26：** The split frequency for input audio is 170Hz. The split frequency for output noise reduce is 165Hz.<br>
