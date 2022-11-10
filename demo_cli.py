import argparse
from ctypes import alignment
import os
from pathlib import Path
import spacy
import matplotlib.pyplot as plt

import librosa
import numpy as np
import soundfile as sf
import torch
import noisereduce as nr  

from encoder import inference as encoder
from encoder.params_data import *
from encoder.params_model import model_embedding_size as speaker_embedding_size
from synthesizer.inference import Synthesizer
from utils.argutils import print_args
from utils.default_models import ensure_default_models
from vocoder import inference as vocoder
from vocoder.display import save_attention
from fixSpeed import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--run_id", type=str, default="default", help= \
    "Name for this model. By default, training outputs will be stored to saved_models/<run_id>/. If a model state "
    "from the same run ID was previously saved, the training will restart from there. Pass -f to overwrite saved "
    "states and restart from scratch.")
    parser.add_argument("-m", "--models_dir", type=Path, default="saved_models",
                        help="Directory containing all saved models")
    parser.add_argument("--griffin_lim",
                        action="store_true",
                        help="if True, use vocoder, else use griffin-lim")
    parser.add_argument("--cpu", action="store_true", help=\
        "If True, processing is done on CPU, even when a GPU is available.")
    parser.add_argument("--no_sound", action="store_true", help=\
        "If True, audio won't be played.")
    parser.add_argument("--seed", type=int, default=None, help=\
        "Optional random number seed value to make toolbox deterministic.")
    args = parser.parse_args()
    arg_dict = vars(args)
    print_args(args, parser)

    # Hide GPUs from Pytorch to force CPU processing
    if arg_dict.pop("cpu"):
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    print("Running a test of your configuration...\n")

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
    ensure_default_models(args.run_id, Path("saved_models"))
    encoder.load_model(list(args.models_dir.glob(f"{args.run_id}/encoder.pt"))[0])
    synthesizer = Synthesizer(list(args.models_dir.glob(f"{args.run_id}/synthesizer.pt"))[0])
    if not args.griffin_lim:
        vocoder.load_model(list(args.models_dir.glob(f"{args.run_id}/vocoder.pt"))[0])


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
    weight = 1 # 声音美颜的用户语音权重
    amp = 1

    while True:
        try:
            # Get the reference audio filepath
            message = "Reference voice: enter an audio filepath of a voice to be cloned (mp3, " \
                      "wav, m4a, flac, ...):\n"
            in_fpath = Path(input(message).replace("\"", "").replace("\'", ""))
            fpath_without_ext = os.path.splitext(str(in_fpath))[0]
            speaker_name = os.path.normpath(fpath_without_ext).split(os.sep)[-1]

            ## Computing the embedding
            # First, we load the wav using the function that the speaker encoder provides. This is
            # important: there is preprocessing that must be applied.

            # The following two methods are equivalent:
            # - Directly load from the filepath:
            # preprocessed_wav = encoder.preprocess_wav(in_fpath)
            # - If the wav is already loaded:

            # get duration info from input audio
            is_wav_file, wav, wav_path = TransFormat(in_fpath, 'wav')
            # 除了m4a格式无法工作而必须转换以外，无论原格式是否为wav，从稳定性的角度考虑也最好再转为wav（因为某些wav本身不带比特率属性，无法在此代码中工作，因此需要转换以赋予其该属性）
            path_ori, filename_ori = os.path.split(wav_path)
            totDur_ori, nPause_ori, arDur_ori, nSyl_ori, arRate_ori = AudioAnalysis(path_ori, filename_ori)
            DelFile(path_ori, '.TextGrid')

            if not is_wav_file:
                os.remove(wav_path)  # remove intermediate wav files
            
            preprocessed_wav = encoder.preprocess_wav(wav)

            print("Loaded input audio file succesfully")

            # Then we derive the embedding. There are many functions and parameters that the
            # speaker encoder interfaces. These are mostly for in-depth research. You will typically
            # only use this function (with its default parameters):
            input_embed = encoder.embed_utterance(preprocessed_wav)

            # Choose standard audio

            fft_max_freq = vocoder.get_dominant_freq(preprocessed_wav)
            print(f"\nthe dominant frequency of input audio is {fft_max_freq}Hz")
            if fft_max_freq < split_freq:
                standard_fpath = "standard_audios/male_1.wav"
            else:
                standard_fpath = "standard_audios/female_1.wav"

            standard_wav = Synthesizer.load_preprocess_wav(standard_fpath)
            preprocessed_standard_wav = encoder.preprocess_wav(standard_wav)
            print("Loaded standard audio file succesfully")

            standard_embed = encoder.embed_utterance(preprocessed_standard_wav)

            embed1=input_embed.dot(weight)
            embed2=standard_embed.dot(1 - weight)
            embed=embed1+embed2

            embed[embed < set_zero_thres]=0 # 噪声值置零
            embed = embed * amp

            ## Generating the spectrogram
            text = input("Write a sentence to be synthesized:\n")

            # If seed is specified, reset torch seed and force synthesizer reload
            if args.seed is not None:
                torch.manual_seed(args.seed)
                synthesizer = Synthesizer(args.syn_model_fpath)
            
            import re
            # The synthesizer works in batch, so you need to put your data in a list or numpy array
            def split_text(text):
                text = text.replace('-', ' ')
                text = text.replace(',', '.')
                text = text.replace(';', '.')
                text = text.replace(':', '.')
                def convert(match_obj):
                    if match_obj.group() is not None:
                        return match_obj.group().replace('.', ',')
                    
                text = re.sub(r"[0-9]+[\.][0-9]+", convert, text)
                texts = [i.text.strip() for i in nlp(text).sents]  # split paragraph to sentences
                return texts
            texts = split_text(text)
            print(f"the list of inputs texts:\n{texts}")

            embeds = [embed] * len(texts)
            # If you know what the attention layer alignments are, you can retrieve them here by
            # passing return_alignments=True
            specs, alignments = synthesizer.synthesize_spectrograms(texts, embeds, return_alignments=True)
            save_attention(alignments.detach().cpu().numpy()[-1, :, :], "attention")

            breaks = [spec.shape[1] for spec in specs]
            spec = np.concatenate(specs, axis=1)
            print("Created the mel spectrogram")


            ## Generating the waveform
            print("Synthesizing the waveform:")

            # If seed is specified, reset torch seed and reload vocoder
            if args.seed is not None:
                torch.manual_seed(args.seed)
                vocoder.load_model(args.voc_model_fpath)

            # Synthesizing the waveform is fairly straightforward. Remember that the longer the
            # spectrogram, the more time-efficient the vocoder.
            if not args.griffin_lim:
                wav = vocoder.infer_waveform(spec)
            else:
                wav = Synthesizer.griffin_lim(spec)

            wav = vocoder.waveform_denoising(wav)

            # Add breaks
            b_ends = np.cumsum(np.array(breaks) * Synthesizer.hparams.hop_size)
            b_starts = np.concatenate(([0], b_ends[:-1]))
            wavs = [wav[start:end] for start, end, in zip(b_starts, b_ends)]
            breaks = [np.zeros(int(0.15 * Synthesizer.sample_rate))] * len(breaks)
            wav = np.concatenate([i for w, b in zip(wavs, breaks) for i in (w, b)])

            # Trim excess silences to compensate for gaps in spectrograms (issue #53)
            # generated_wav = encoder.preprocess_wav(generated_wav)
            wav = wav / np.abs(wav).max() * 1

            # Save it on the disk
            # filename = "demo_output_%02d.wav" % num_generated
            if not os.path.exists("out_audios"):
                os.mkdir("out_audios")
            filename = f"out_audios/{speaker_name}_syn.wav"
            # print(wav.dtype)
            sf.write(filename, wav.astype(np.float32), synthesizer.sample_rate)
            num_generated += 1
            print("\nSaved output (havent't change speed) as %s\n\n" % filename)

            # Fix Speed(generate new audio)
            fix_file = work(totDur_ori, 
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


        except Exception as e:
            print("Caught exception: %s" % repr(e))
            print("Restarting\n")
