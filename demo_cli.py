import argparse
from ctypes import alignment
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from pathlib import Path
import spacy
import time


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--run_id", type=str, default="default_emotion", help= \
    "Name for this model. By default, training outputs will be stored to saved_models/<run_id>/. If a model state "
    "from the same run ID was previously saved, the training will restart from there. Pass -f to overwrite saved "
    "states and restart from scratch.")
    parser.add_argument("-m", "--models_dir", type=Path, default="saved_models",
                        help="Directory containing all saved models")
    parser.add_argument("--emotion_encoder_model_fpath", type=Path,
                        default="saved_models/default_emotion/INTERSECT_46_dilation_8_dropout_05_add_esd_npairLoss", help=\
        "Path your trained emotion encoder model.")
    parser.add_argument("--weight", type=float, default=1,
                        help="weight of input audio for voice filter")
    parser.add_argument("--griffin_lim",
                        action="store_true",
                        help="if True, use griffin-lim, else use vocoder")
    parser.add_argument("--cpu", action="store_false", help=\
        "If True, processing is done on CPU, even when a GPU is available.")
    parser.add_argument("--no_sound", action="store_true", help=\
        "If True, audio won't be played.")
    parser.add_argument("--seed", type=int, default=None, help=\
        "Optional random number seed value to make toolbox deterministic.")
    args = parser.parse_args()
    arg_dict = vars(args)
    # print_args(args, parser)

    # Hide GPUs from Pytorch to force CPU processing
    if arg_dict.pop("cpu"):
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    print("Running a test of your configuration...\n")
    
    import numpy as np
    import soundfile as sf
    import torch
    import numpy as np
    import os
    import tensorflow as tf
    from emotion_encoder.Model import TIMNET_Model
    import argparse
    import json

    import speaker_encoder.inference
    import speaker_encoder.params_data 
    from emotion_encoder.utils import get_mfcc
    from synthesizer.inference import Synthesizer_infer
    from synthesizer.utils.cleaners import add_breaks, english_cleaners_predict
    from vocoder import inference as vocoder
    from vocoder.display import save_attention_multiple, save_spectrogram, save_stop_tokens
    from utils.argutils import print_args
    from utils.default_models import ensure_default_models
    from speed_changer.fixSpeed import *
    if torch.cuda.is_available():
        device_id = torch.cuda.current_device()
        gpu_properties = torch.cuda.get_device_properties(device_id)
        ## Print some environment information (for debugging purposes)
        print("Found %d GPUs available. Using GPU %d (%s) of compute capability %d.%d with "
            "%.1fGb total memory.\n" %
            (torch.cuda.device_count(),
            device_id,
            gpu_properties.name,
            gpu_properties.major,
            gpu_properties.minor,
            gpu_properties.total_memory / 1e9))
    else:
        print("Using CPU for inference.\n")

    ## Load the models one by one.
    if not args.griffin_lim:
        print("Preparing the encoder, the synthesizer and the vocoder...")
    else:
        print("Preparing the encoder and the synthesizer...")
    ensure_default_models(args.run_id, args.models_dir)  #找到模型，如果没找到就在网络上下载默认模型保证有模型可用
    speaker_encoder.inference.load_model(list(args.models_dir.glob(f"{args.run_id}/encoder.pt"))[0])
    synthesizer = Synthesizer_infer(list(args.models_dir.glob(f"{args.run_id}/synthesizer.pt"))[0], model_name="EmotionTacotron")
    if not args.griffin_lim:
        vocoder.load_model(list(args.models_dir.glob(f"{args.run_id}/vocoder.pt"))[0])

    ### prepare the emotion encoder ###
    json_fpath = os.path.join(args.emotion_encoder_model_fpath, "params.json")
    f = open(json_fpath)
    emotion_encoder_args = argparse.Namespace(**json.load(f))
        
    os.environ['CUDA_VISIBLE_DEVICES'] = emotion_encoder_args.gpu
    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth=True
    session = tf.compat.v1.Session(config=config)
    print(f"###gpus:{gpus}")

    CLASS_LABELS = ("angry", "happy", "neutral", "sad", "surprise")

    emotion_encoder = TIMNET_Model(args=emotion_encoder_args, input_shape=(626, 39), class_label=CLASS_LABELS)

    emotion_encoder.create_model()

    # ## Run a test
    # print("Testing your configuration with small inputs.")
    # # Forward an audio waveform of zeroes that lasts 1 second. Notice how we can get the encoder's
    # # sampling rate, which may differ.
    # # If you're unfamiliar with digital audio, know that it is encoded as an array of floats
    # # (or sometimes integers, but mostly floats in this projects) ranging from -1 to 1.
    # # The sampling rate is the number of values (samples) recorded per second, it is set to
    # # 16000 for the encoder. Creating an array of length <sampling_rate> will always correspond
    # # to an audio of 1 second.
    # print("\tTesting the encoder...")
    # encoder.embed_utterance(np.zeros(encoder.sampling_rate))

    # # Create a dummy embedding. You would normally use the embedding that encoder.embed_utterance
    # # returns, but here we're going to make one ourselves just for the sake of showing that it's
    # # possible.
    # embed = np.random.rand(speaker_embedding_size)
    # # Embeddings are L2-normalized (this isn't important here, but if you want to make your own
    # # embeddings it will be).
    # embed /= np.linalg.norm(embed)
    # # The synthesizer can handle multiple inputs with batching. Let's create another embedding to
    # # illustrate that
    # embeds = [embed, np.zeros(speaker_embedding_size)]
    # texts = ["test 1", "test 2"]
    # print("\tTesting the synthesizer... (loading the model will output a lot of text)")
    # mels = synthesizer.synthesize_spectrograms(texts, embeds)

    # # The vocoder synthesizes one waveform at a time, but it's more efficient for long ones. We
    # # can concatenate the mel spectrograms to a single one.
    # mel = np.concatenate(mels, axis=1)
    # # The vocoder can take a callback function to display the generation. More on that later. For
    # # now we'll simply hide it like this:
    # if not args.griffin_lim:
    #     no_action = lambda *args: None
    #     print("\tTesting the vocoder...")
    #     # For the sake of making this test short, we'll pass a short target length. The target length
    #     # is the length of the wav segments that are processed in parallel. E.g. for audio sampled
    #     # at 16000 Hertz, a target length of 8000 means that the target audio will be cut in chunks of
    #     # 0.5 seconds which will all be generated together. The parameters here are absurdly short, and
    #     # that has a detrimental effect on the quality of the audio. The default parameters are
    #     # recommended in general.
    #     vocoder.infer_waveform(mel, target=200, overlap=50, progress_callback=no_action)

    # print("All test passed! You can now synthesize speech.\n\n")


    ## Interactive speech generation
    print("This is a GUI-less example of interface to SV2TTS. The purpose of this script is to "
          "show how you can interface this project easily with your own. See the source code for "
          "an explanation of what is happening.\n")

    print("Interactive generation loop")
    num_generated = 0

    nlp = spacy.load('en_core_web_sm')
    weight = arg_dict["weight"] # 声音美颜的用户语音权重
    amp = 1

    while True:
        # try:
        # Get the reference audio filepath

        # Computing the embedding
        # First, we load the wav using the function that the speaker encoder provides. This is
        # important: there is preprocessing that must be applied.

        # The following two methods are equivalent:
        # - Directly load from the filepath:
        # preprocessed_wav = encoder.preprocess_wav(in_fpath)
        # - If the wav is already loaded:

        # get duration info from input audio
        message2 = "Reference voice: enter an audio folder of a voice to be cloned (mp3, " \
                f"wav, m4a, flac, ...):\n"
        in_fpath = Path(input(message2).replace("\"", "").replace("\'", ""))


        fpath_without_ext = os.path.splitext(str(in_fpath))[0]
        speaker_name = os.path.normpath(fpath_without_ext).split(os.sep)[-1]

        is_wav_file, single_wav, wav_path = TransFormat(in_fpath, 'wav')

        if not is_wav_file:
            os.remove(wav_path)  # remove intermediate wav files
        # # merge
        # if i == 0:
        wav = single_wav
        # else:
        #     wav = np.append(wav, single_wav)
        # write to disk
        path_ori, _ = os.path.split(wav_path)
        file_ori = 'temp.wav'
        fpath = os.path.join(path_ori, file_ori)
        sf.write(fpath, wav, samplerate=speaker_encoder.params_data.sampling_rate)

        # adjust the speed
        totDur_ori, nPause_ori, arDur_ori, nSyl_ori, arRate_ori = AudioAnalysis(path_ori, file_ori)
        DelFile(path_ori, '.TextGrid')
        os.remove(fpath)

        preprocessed_wav = speaker_encoder.inference.preprocess_wav(wav)

        print("Loaded input audio file successfully")

        # Then we derive the embedding. There are many functions and parameters that the
        # speaker encoder interfaces. These are mostly for in-depth research. You will typically
        # only use this function (with its default parameters):
        speaker_embed = speaker_encoder.inference.embed_utterance(preprocessed_wav)
        mfcc = get_mfcc(wav, syn_hparams.sample_rate, mean_signal_length=320000)
        emotion_embed = emotion_encoder.infer(np.array([mfcc]), model_dir=args.emotion_encoder_model_fpath)[0]

        # Choose standard audio
        # fft_max_freq = vocoder.get_dominant_freq(preprocessed_wav)
        # print(f"\nthe dominant frequency of input audio is {fft_max_freq}Hz")
        # if fft_max_freq < speaker_encoder.params_data.split_freq:
        #     vocoder.hp.sex = 1
        #     standard_fpath = "standard_audios/male_1.wav"
        # else:
        #     vocoder.hp.sex = 0
        #     standard_fpath = "standard_audios/female_1.wav"

        # if os.path.exists(standard_fpath):
            
        #     standard_wav = Synthesizer.load_preprocess_wav(standard_fpath)
        #     preprocessed_standard_wav = speaker_encoder.inference.preprocess_wav(standard_wav)
        #     print("Loaded standard audio file successfully")

        #     standard_embed = speaker_encoder.inference.embed_utterance(preprocessed_standard_wav)

        #     embed1=np.copy(speaker_embed).dot(weight)
        #     embed2=np.copy(standard_embed).dot(1 - weight)
        #     embed=embed1+embed2
        # else: 
        #     embed = np.copy(speaker_embed)

        # embed[embed < speaker_encoder.params_data.set_zero_thres]=0 # 噪声值置零
        # embed = embed * amp

        start_syn = time.time()
        # Generating the spectrogram
        # text = input("Write a sentence to be synthesized:\n")
        text = "We have to reduce the number of plastic bags."

        # If seed is specified, reset torch seed and force synthesizer reload
        if args.seed is not None:
            torch.manual_seed(args.seed)
            synthesizer = Synthesizer_infer(args.syn_model_fpath)

        # The synthesizer works in batch, so you need to put your data in a list or numpy array
        def preprocess_text(text):
            text = add_breaks(text) 
            text = english_cleaners_predict(text)
            texts = [i.text.strip() for i in nlp(text).sents]  # split paragraph to sentences
            return texts

        texts = preprocess_text(text)
        print(f"the list of inputs texts:\n{texts}")

        specs = []
        alignments = []
        stop_tokens = []

        for text in texts:
            spec, align, stop_token = synthesizer.synthesize_spectrograms([text], [speaker_embed], [emotion_embed], require_visualization=True)
            specs.append(spec[0])
            alignments.append(align[0])
            stop_tokens.append(stop_token[0])

        breaks = [spec.shape[1] for spec in specs]
        spec = np.concatenate(specs, axis=1)
        alignments = np.array(alignments)
        stop_tokens = np.array(stop_tokens)
        

        ## Save synthesizer visualization results
        if not os.path.exists("syn_results"):
            os.mkdir("syn_results")
        save_attention_multiple(alignments, "syn_results/attention")
        save_stop_tokens(stop_tokens, "syn_results/stop_tokens")
        save_spectrogram(spec, "syn_results/mel")
        print("Created the mel spectrogram")

        end_syn = time.time()
        print(f"Prediction time of synthesizer is {end_syn - start_syn}s")

        start_voc = time.time()
        ## Generating the waveform
        print("Synthesizing the waveform:")

        # If seed is specified, reset torch seed and reload vocoder
        if args.seed is not None:
            torch.manual_seed(args.seed)
            vocoder.load_model(args.voc_model_fpath)

        # Synthesizing the waveform is fairly straightforward. Remember that the longer the
        # spectrogram, the more time-efficient the vocoder.
        if not args.griffin_lim:
            wav = vocoder.infer_waveform(spec, target=vocoder.hp.voc_target, overlap=vocoder.hp.voc_overlap, crossfade=vocoder.hp.is_crossfade) 
        else:
            wav = Synthesizer_infer.griffin_lim(spec)

        end_voc = time.time()
        print(f"Prediction time of vocoder is {end_voc - start_voc}s")
        print(f"Prediction time of TTS is {end_voc - start_syn}s")

        # Add breaks
        b_ends = np.cumsum(np.array(breaks) * Synthesizer_infer.hparams.hop_size)
        b_starts = np.concatenate(([0], b_ends[:-1]))
        wavs = [wav[start:end] for start, end, in zip(b_starts, b_ends)]
        breaks = [np.zeros(int(0.15 * Synthesizer_infer.sample_rate))] * len(breaks)
        wav = np.concatenate([i for w, b in zip(wavs, breaks) for i in (w, b)])

        # Trim excess silences to compensate for gaps in spectrograms (issue #53)
        # generated_wav = encoder.inference.preprocess_wav(wav)
        wav = wav / np.abs(wav).max() * 4

        # Save it on the disk
        # filename = "demo_output_%02d.wav" % num_generated
        if not os.path.exists("out_audios"):
            os.mkdir("out_audios")
        
        dir_path = os.path.dirname(os.path.realpath(__file__))  # current dir 
        filename = os.path.join(dir_path, f"out_audios/{speaker_name}_syn.wav")
        # print(wav.dtype)
        sf.write(filename, wav.astype(np.float32), synthesizer.sample_rate)
        num_generated += 1
        print("\nSaved output (haven't change speed) as %s\n\n" % filename)

        # Fix Speed(generate new audio)
        fix_file, speed_factor = work(totDur_ori, 
                        nPause_ori, 
                        arDur_ori, 
                        nSyl_ori, 
                        arRate_ori, 
                        filename)
        print(f"\nSaved output (fixed speed) as {fix_file}\n\n")


        # # Play the audio (non-blocking)
        # if not args.no_sound:
        #     import sounddevice as sd
        #     try:
        #         sd.stop()
        #         sd.play(wav, synthesizer.sample_rate)
        #     except sd.PortAudioError as e:
        #         print("\nCaught exception: %s" % repr(e))
        #         print("Continuing without audio playback. Suppress this message with the \"--no_sound\" flag.\n")
        #     except:
        #         raise


        # except Exception as e:
        #     print("Caught exception: %s" % repr(e))
        #     print("Restarting\n")
