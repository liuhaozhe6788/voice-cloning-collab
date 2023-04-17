import argparse
from ctypes import alignment
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from pathlib import Path
import spacy
import time


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
    parser.add_argument("--weight", type=float, default=0.7,
                        help="weight of input audio for voice filter")
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
    # print_args(args, parser)

    # Hide GPUs from Pytorch to force CPU processing
    if arg_dict.pop("cpu"):
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    print("Running a test of your configuration...\n")

    import librosa
    import numpy as np
    import soundfile as sf
    import torch
    import noisereduce as nr  

    import encoder.inference
    import encoder.params_data 
    from synthesizer.inference import Synthesizer
    from synthesizer.utils.cleaners import add_breaks, english_cleaners
    from vocoder import inference as vocoder
    from vocoder.display import save_attention, save_spectrogram, save_stop_tokens
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
    ensure_default_models(args.run_id, Path("saved_models"))
    encoder.inference.load_model(list(args.models_dir.glob(f"{args.run_id}/encoder.pt"))[0])
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
    weight = arg_dict["weight"] # 声音美颜的用户语音权重
    amp = 1

    # while True:
    # try:
    # Get the reference audio filepath
    while True:
        # enter the number of reference audios
        # message1 = "Please enter the number of reference audios:\n"
        # num_of_input_audio = int(input(message1))
        num_of_input_audio = 1

        for i in range(num_of_input_audio):
            # Computing the embedding
            # First, we load the wav using the function that the speaker encoder provides. This is
            # important: there is preprocessing that must be applied.

            # The following two methods are equivalent:
            # - Directly load from the filepath:
            # preprocessed_wav = encoder.preprocess_wav(in_fpath)
            # - If the wav is already loaded:

            # get duration info from input audio
            message2 = "Reference voice: enter an audio folder of a voice to be cloned (mp3, " \
                       f"wav, m4a, flac, ...):({i+1}/{num_of_input_audio})\n"
            in_fpath = Path(input(message2).replace("\"", "").replace("\'", ""))
            # in_fpath = Path("/home/liuhaozhe/voice_cloning_project/collected_audios/celeb_audios/trimmed/Madonna_trim.wav")

            fpath_without_ext = os.path.splitext(str(in_fpath))[0]
            speaker_name = os.path.normpath(fpath_without_ext).split(os.sep)[-1]

            is_wav_file, single_wav, wav_path = TransFormat(in_fpath, 'wav')
            # 除了m4a格式无法工作而必须转换以外，无论原格式是否为wav，从稳定性的角度考虑也最好再转为wav（因为某些wav本身不带比特率属性，无法在此代码中工作，因此需要转换以赋予其该属性）

            if not is_wav_file:
                os.remove(wav_path)  # remove intermediate wav files
            # merge
            if i == 0:
                wav = single_wav
            else:
                wav = np.append(wav, single_wav)
        # write to disk
        path_ori, _ = os.path.split(wav_path)
        file_ori = 'temp.wav'
        fpath = os.path.join(path_ori, file_ori)
        sf.write(fpath, wav, samplerate=encoder.params_data.sampling_rate)

        # adjust the speed
        totDur_ori, nPause_ori, arDur_ori, nSyl_ori, arRate_ori = AudioAnalysis(path_ori, file_ori)
        DelFile(path_ori, '.TextGrid')
        os.remove(fpath)

        preprocessed_wav = encoder.inference.preprocess_wav(wav)

        print("Loaded input audio file succesfully")

        # Then we derive the embedding. There are many functions and parameters that the
        # speaker encoder interfaces. These are mostly for in-depth research. You will typically
        # only use this function (with its default parameters):
        input_embed = encoder.inference.embed_utterance(preprocessed_wav)
        # Choose standard audio

        fft_max_freq = vocoder.get_dominant_freq(preprocessed_wav)
        print(f"\nthe dominant frequency of input audio is {fft_max_freq}Hz")
        if fft_max_freq < encoder.params_data.split_freq:
            standard_fpath = "standard_audios/male_1.wav"
        else:
            standard_fpath = "standard_audios/female_1.wav"

        if os.path.exists(standard_fpath):
            
            standard_wav = Synthesizer.load_preprocess_wav(standard_fpath)
            preprocessed_standard_wav = encoder.inference.preprocess_wav(standard_wav)
            print("Loaded standard audio file succesfully")

            standard_embed = encoder.inference.embed_utterance(preprocessed_standard_wav)

            embed1=np.copy(input_embed).dot(weight)
            embed2=np.copy(standard_embed).dot(1 - weight)
            embed=embed1+embed2
        else: 
            embed = np.copy(input_embed)

        embed[embed < encoder.params_data.set_zero_thres]=0 # 噪声值置零
        embed = embed * amp

        start_syn = time.time()
        # Generating the spectrogram
        text = input("Write a sentence to be synthesized:\n")
        # text = "Introduction to Mechanics:"\
        #     "Mechanics is a branch of physics that deals with the behavior of physical bodies under the influence of various forces. The study of mechanics is important in understanding the behavior of machines, the motion of objects, and the principles of engineering. Mechanics has been an essential part of physics since ancient times and has continued to evolve with advancements in science and technology. This paper will discuss the principles of mechanics, the laws of motion, and the applications of mechanics in engineering and technology."\
        #     " "\
        #     "Principles of Mechanics:"\
        #     " "\
        #     "The principles of mechanics are based on the laws of motion, which were first proposed by Sir Isaac Newton in the 17th century. The laws of motion form the foundation of mechanics and provide a framework for understanding the behavior of physical bodies. The three laws of motion are:"\
        #     " "\
        #     "Law of Inertia: An object at rest will remain at rest, and an object in motion will continue to move in a straight line at a constant speed unless acted upon by an external force."\
        #     " "\
        #     "Law of Acceleration: The acceleration of an object is directly proportional to the force applied to it and inversely proportional to its mass."\
        #     " "\
        #     "Law of Action and Reaction: For every action, there is an equal and opposite reaction."\
        #     " "\
        #     "These laws of motion are essential for understanding the behavior of objects in motion and for predicting the motion of objects in various situations."\
        #     " "\
        #     "Applications of Mechanics:"\
        #     " "\
        #     "Mechanics has many practical applications in engineering and technology. Some of these applications include:"\
        #     " "\
        #     "1.Aerospace Engineering: Mechanics is essential in the design and construction of aircraft and spacecraft. The principles of mechanics are used to calculate the trajectory of a spacecraft, determine the forces acting on it, and ensure that it remains stable during flight."\
        #     " "\
        #     "2.Civil Engineering: Mechanics is used in the design and construction of bridges, buildings, and other structures. The principles of mechanics are used to calculate the forces acting on a structure, determine its stability, and ensure that it can withstand various forces, such as wind and earthquakes."\
        #     " "\
        #     "3.Automotive Engineering: Mechanics is used in the design and construction of automobiles. The principles of mechanics are used to determine the forces acting on a vehicle, calculate its speed and acceleration, and ensure that it remains stable during operation."\
        #     " "\
        #     "4.Robotics: Mechanics is used in the design and construction of robots. The principles of mechanics are used to calculate the forces acting on a robot, determine its stability, and ensure that it can perform its tasks safely and efficiently."\
        #     " "\
        #     "5.Manufacturing: Mechanics is used in the design and operation of machines used in manufacturing processes. The principles of mechanics are used to ensure that machines operate safely and efficiently, and to optimize their performance."\
        #     " "\
        #     "Conclusion:"\
        #     " "\
        #     "Mechanics is an essential branch of physics that provides a framework for understanding the behavior of physical bodies under the influence of various forces. The principles of mechanics are based on the laws of motion, which form the foundation of the field. Mechanics has many practical applications in engineering and technology, from aerospace and automotive engineering to robotics and manufacturing. As science and technology continue to evolve, the principles of mechanics will remain an important part of our understanding of the physical world."

        # If seed is specified, reset torch seed and force synthesizer reload
        if args.seed is not None:
            torch.manual_seed(args.seed)
            synthesizer = Synthesizer(args.syn_model_fpath)

        # The synthesizer works in batch, so you need to put your data in a list or numpy array
        def preprocess_text(text):
            text = add_breaks(text) 
            text = english_cleaners(text)
            texts = [i.text.strip() for i in nlp(text).sents]  # split paragraph to sentences
            return texts

        texts = preprocess_text(text)
        print(f"the list of inputs texts:\n{texts}")

        # embeds = [embed] * len(texts)
        specs = []
        alignments = []
        stop_tokens = []
        for i, text in enumerate(texts):
            print(f"No.{i} sequence is {text}")
            spec, align, stop_token = synthesizer.synthesize_spectrograms([text], [embed], require_visualization=True)
            specs.append(spec[0])
            alignments.append(align[0])
            stop_tokens.append(stop_token[0])

        breaks = [spec.shape[1] for spec in specs]
        spec = np.concatenate(specs, axis=1)
        

        ## Save synthesizer visualization results
        if not os.path.exists("syn_results"):
            os.mkdir("syn_results")
        save_attention(alignments, "syn_results/attention")
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
            wav = vocoder.infer_waveform(spec)
        else:
            wav = Synthesizer.griffin_lim(spec)
        wav = vocoder.waveform_denoising(wav)
        end_voc = time.time()
        print(f"Prediction time of vocoder is {end_voc - start_voc}s")
        print(f"Prediction time of TTS is {end_voc - start_syn}s")

        # Add breaks
        b_ends = np.cumsum(np.array(breaks) * Synthesizer.hparams.hop_size)
        b_starts = np.concatenate(([0], b_ends[:-1]))
        wavs = [wav[start:end] for start, end, in zip(b_starts, b_ends)]
        breaks = [np.zeros(int(0.15 * Synthesizer.sample_rate))] * len(breaks)
        wav = np.concatenate([i for w, b in zip(wavs, breaks) for i in (w, b)])

        # Trim excess silences to compensate for gaps in spectrograms (issue #53)
        # generated_wav = encoder.preprocess_wav(generated_wav)
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


        # except Exception as e:
        #     print("Caught exception: %s" % repr(e))
        #     print("Restarting\n")
