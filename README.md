# voice-cloning
## Version updates
**2022.05.19：** I calculated GE2E loss in encoder with cuda rather than originally-configured CPU. It speeds up the encoder training speed.<br>
**2022.07.15：** I added Loss animation plot for synthesizer and vocoder.<br>
**2022.07.19：** I added response time and Griffin-Lim vocoder results for demo_toolbox.<br>
**2022.07.29：** I added model validation for encoder, synthesizer and vocoder.<br>
**2022.08.02：** I added voxceleb train and dev data for encoder. I added noise reduce method for the output wav from vocoder.<br>
noisereduce reference: https://github.com/timsainb/noisereduce<br>
**2022.08.06：** I split the long text into short sentences using spacy for input of synthesizer.
