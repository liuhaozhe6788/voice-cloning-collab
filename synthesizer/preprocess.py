from multiprocessing.pool import Pool
from synthesizer import audio
from functools import partial
from itertools import chain, groupby
from speaker_encoder import inference as speaker_encoder
from pathlib import Path
from utils import logmmse
from tqdm import tqdm
import numpy as np
import librosa
import random
from emotion_encoder.utils import get_mfcc


def preprocess_librispeech(datasets_root: Path, out_dir: Path, n_processes: int, skip_existing: bool, hparams,
                       datasets_name: str, subfolders: str, no_alignments=False):
    
    # Gather the input directories of LibriSpeeech
    dataset_root = datasets_root.joinpath(datasets_name)
    input_dirs = [dataset_root.joinpath(subfolder.strip()) for subfolder in subfolders.split(",")]
    print("\n    ".join(map(str, ["Using data from:"] + input_dirs)))
    assert all(input_dir.exists() for input_dir in input_dirs)
    
    train_input_dirs = input_dirs[: -1]
    dev_input_dirs = input_dirs[-1: ]

    # Create the output directories for each output file type
    train_out_dir = out_dir.joinpath("train")
    train_out_dir.mkdir(exist_ok=True)
    train_out_dir.joinpath("mels").mkdir(exist_ok=True)
    train_out_dir.joinpath("audio").mkdir(exist_ok=True)
    train_out_dir.joinpath("mfccs").mkdir(exist_ok=True)
    
    # Create a metadata file
    train_metadata_fpath = train_out_dir.joinpath("train.txt")
    train_metadata_file = train_metadata_fpath.open("a" if skip_existing else "w", encoding="utf-8")
    
    dev_out_dir = out_dir.joinpath("dev")
    dev_out_dir.mkdir(exist_ok=True)
    dev_out_dir.joinpath("mels").mkdir(exist_ok=True)
    dev_out_dir.joinpath("audio").mkdir(exist_ok=True)
    dev_out_dir.joinpath("mfccs").mkdir(exist_ok=True)

    # Create a metadata file
    dev_metadata_fpath = dev_out_dir.joinpath("dev.txt")
    dev_metadata_file = dev_metadata_fpath.open("a" if skip_existing else "w", encoding="utf-8")

    # Preprocess the train dataset
    train_speaker_dirs = list(chain.from_iterable(train_input_dir.glob("*") for train_input_dir in train_input_dirs))
    func = partial(preprocess_speaker, out_dir=train_out_dir, skip_existing=skip_existing,
                   hparams=hparams, no_alignments=no_alignments)
    job = Pool(n_processes).imap(func, train_speaker_dirs)
    for speaker_metadata in tqdm(job, datasets_name, len(train_speaker_dirs), unit="speakers"):
        for metadatum in speaker_metadata:
            train_metadata_file.write("|".join(str(x) for x in metadatum) + "\n")
    train_metadata_file.close()

    # Verify the contents of the metadata file
    with train_metadata_fpath.open("r", encoding="utf-8") as train_metadata_file:
        metadata = [line.split("|") for line in train_metadata_file]
    mel_frames = sum([int(m[4]) for m in metadata])
    timesteps = sum([int(m[3]) for m in metadata])
    sample_rate = hparams.sample_rate
    hours = (timesteps / sample_rate) / 3600
    print("The train dataset consists of %d utterances, %d mel frames, %d audio timesteps (%.2f hours)." %
          (len(metadata), mel_frames, timesteps, hours))
    print("Max input length (text chars): %d" % max(len(m[5]) for m in metadata))
    print("Max mel frames length: %d" % max(int(m[4]) for m in metadata))
    print("Max audio timesteps length: %d" % max(int(m[3]) for m in metadata))
    
    # Preprocess the dev dataset
    dev_speaker_dirs = list(chain.from_iterable(dev_input_dir.glob("*") for dev_input_dir in dev_input_dirs))
    func = partial(preprocess_speaker, out_dir=dev_out_dir, skip_existing=skip_existing,
                   hparams=hparams, no_alignments=no_alignments)
    job = Pool(n_processes).imap(func, dev_speaker_dirs)
    for speaker_metadata in tqdm(job, datasets_name, len(dev_speaker_dirs), unit="speakers"):
        for metadatum in speaker_metadata:
            dev_metadata_file.write("|".join(str(x) for x in metadatum) + "\n")
    dev_metadata_file.close()

    # Verify the contents of the metadata file
    with dev_metadata_fpath.open("r", encoding="utf-8") as dev_metadata_file:
        metadata = [line.split("|") for line in dev_metadata_file]
    mel_frames = sum([int(m[4]) for m in metadata])
    timesteps = sum([int(m[3]) for m in metadata])
    sample_rate = hparams.sample_rate
    hours = (timesteps / sample_rate) / 3600
    print("The dev dataset consists of %d utterances, %d mel frames, %d audio timesteps (%.2f hours)." %
          (len(metadata), mel_frames, timesteps, hours))
    print("Max input length (text chars): %d" % max(len(m[5]) for m in metadata))
    print("Max mel frames length: %d" % max(int(m[4]) for m in metadata))
    print("Max audio timesteps length: %d" % max(int(m[3]) for m in metadata))


def preprocess_vctk(datasets_root: Path, out_dir: Path, n_processes: int, skip_existing: bool, hparams,
                       datasets_name: str, subfolders: str, no_alignments=True):
    dataset_root = datasets_root.joinpath(datasets_name)
    input_dir = dataset_root.joinpath(subfolders)
    print("Using data from:" + str(input_dir))
    assert input_dir.exists()
    paths = [*input_dir.rglob("*.flac")]

    # train dev audio data split
    train_input_fpaths = []
    dev_input_fpaths = []

    pairs = sorted([(p.parts[-2].split('_')[0], p) for p in paths])
    del paths

    for _, group in groupby(pairs, lambda pair: pair[0]):
        paths = sorted([p for _, p in group if "mic1.flac" in str(p)])  # only get mic1 flac file
        random.seed(0)
        random.shuffle(paths)
        n = round(len(paths) * 0.9)
        train_input_fpaths.extend(paths[:n])  
        # dev dataset has the same speakers as train dataset      
        dev_input_fpaths.extend(paths[n:]) 

    # Create the output directories for each output file type
    train_out_dir = out_dir.joinpath("train")
    train_out_dir.mkdir(exist_ok=True)
    train_out_dir.joinpath("mels").mkdir(exist_ok=True)
    train_out_dir.joinpath("audio").mkdir(exist_ok=True)
    train_out_dir.joinpath("mfccs").mkdir(exist_ok=True)
    
    dev_out_dir = out_dir.joinpath("dev")
    dev_out_dir.mkdir(exist_ok=True)
    dev_out_dir.joinpath("mels").mkdir(exist_ok=True)
    dev_out_dir.joinpath("audio").mkdir(exist_ok=True)
    dev_out_dir.joinpath("mfccs").mkdir(exist_ok=True)

    # Preprocess the train dataset
    preprocess_data(train_input_fpaths, mode="train", out_dir=train_out_dir, skip_existing=skip_existing, hparams=hparams, no_alignments=no_alignments)
    
    # Preprocess the dev dataset
    preprocess_data(dev_input_fpaths, mode="dev", out_dir=dev_out_dir, skip_existing=skip_existing, hparams=hparams, no_alignments=no_alignments)


def preprocess_speaker(speaker_dir, out_dir: Path, skip_existing: bool, hparams, no_alignments: bool):
    metadata = []
    for book_dir in speaker_dir.glob("*"):
        if no_alignments:
            # Gather the utterance audios and texts
            # LibriTTS uses .wav but we will include extensions for compatibility with other datasets
            extensions = ["*.wav", "*.flac", "*.mp3"]
            for extension in extensions:
                wav_fpaths = book_dir.glob(extension)

                for wav_fpath in wav_fpaths:
                    # Load the audio waveform
                    wav, _ = librosa.load(str(wav_fpath), hparams.sample_rate)
                    if hparams.rescale:
                        wav = wav / np.abs(wav).max() * hparams.rescaling_max

                    # Get the corresponding text
                    # Check for .txt (for compatibility with other datasets)
                    text_fpath = wav_fpath.with_suffix(".txt")
                    if not text_fpath.exists():
                        # Check for .normalized.txt (LibriTTS)
                        text_fpath = wav_fpath.with_suffix(".normalized.txt")
                        assert text_fpath.exists()
                    with text_fpath.open("r") as text_file:
                        text = "".join([line for line in text_file])
                        text = text.replace("\"", "")
                        text = text.strip()

                    # Process the utterance
                    metadata.append(process_utterance(wav, text, out_dir, str(wav_fpath.with_suffix("").name),
                                                      skip_existing, hparams))
        else:
            # Process alignment file (LibriSpeech support)
            # Gather the utterance audios and texts
            try:
                alignments_fpath = next(book_dir.glob("*.alignment.txt"))
                with alignments_fpath.open("r") as alignments_file:
                    alignments = [line.rstrip().split(" ") for line in alignments_file]
            except StopIteration:
                # A few alignment files will be missing
                continue

            # Iterate over each entry in the alignments file
            for wav_fname, words, end_times in alignments:
                wav_fpath = book_dir.joinpath(wav_fname + ".flac")
                assert wav_fpath.exists()
                words = words.replace("\"", "").split(",")
                end_times = list(map(float, end_times.replace("\"", "").split(",")))

                # Process each sub-utterance
                wavs, texts = split_on_silences(wav_fpath, words, end_times, hparams)
                for i, (wav, text) in enumerate(zip(wavs, texts)):
                    sub_basename = "%s_%02d" % (wav_fname, i)
                    metadata.append(process_utterance(wav, text, out_dir, sub_basename,
                                                      skip_existing, hparams))

    return [m for m in metadata if m is not None]


def preprocess_data(wav_fpaths, mode, out_dir: Path, skip_existing: bool, hparams, no_alignments: bool):
    assert mode in ["train", "dev"]
    # Create a metadata file
    metadata_fpath = out_dir.joinpath(f"{mode}.txt")
    metadata_file = metadata_fpath.open("a", encoding="utf-8")
    if no_alignments:
        for wav_fpath in tqdm(wav_fpaths, desc=mode):
            # Load the audio waveform
            wav, _ = librosa.load(str(wav_fpath), hparams.sample_rate)
            if hparams.rescale:
                wav = wav / np.abs(wav).max() * hparams.rescaling_max

            # Get the corresponding text
            # Check for .txt (for compatibility with other datasets)
            base_name = "_".join(wav_fpath.name.split(".")[0].split("_")[: -1]) + ".txt"
            text_fpath = wav_fpath.with_name(base_name)

            if not text_fpath.exists():
                continue
            with text_fpath.open("r") as text_file:
                text = "".join([line for line in text_file])
                text = text.replace("\"", "")
                text = text.strip()

            # Process the utterance
            metadata = process_utterance(wav, text, out_dir, str(wav_fpath.with_suffix("").name), skip_existing, hparams, trim_silence=False)

            if metadata is not None:
                metadata_file.write("|".join(str(x) for x in metadata) + "\n")
    metadata_file.close()

    # Verify the contents of the metadata file
    with metadata_fpath.open("r", encoding="utf-8") as metadata_file:
        metadata = [line.split("|") for line in metadata_file]
    mel_frames = sum([int(m[4]) for m in metadata])
    timesteps = sum([int(m[3]) for m in metadata])
    sample_rate = hparams.sample_rate
    hours = (timesteps / sample_rate) / 3600
    print(f"The {mode} dataset consists of %d utterances, %d mel frames, %d audio timesteps (%.2f hours)." %
          (len(metadata), mel_frames, timesteps, hours))
    print("Max input length (text chars): %d" % max(len(m[5]) for m in metadata))
    print("Max mel frames length: %d" % max(int(m[4]) for m in metadata))
    print("Max audio timesteps length: %d" % max(int(m[3]) for m in metadata))


def split_on_silences(wav_fpath, words, end_times, hparams):
    # Load the audio waveform
    wav, _ = librosa.load(str(wav_fpath), hparams.sample_rate)
    if hparams.rescale:
        wav = wav / np.abs(wav).max() * hparams.rescaling_max

    words = np.array(words)
    start_times = np.array([0.0] + end_times[:-1])
    end_times = np.array(end_times)
    assert len(words) == len(end_times) == len(start_times)
    assert words[0] == "" and words[-1] == ""

    # Find pauses that are too long
    mask = (words == "") & (end_times - start_times >= hparams.silence_min_duration_split)
    mask[0] = mask[-1] = True
    breaks = np.where(mask)[0]

    # Profile the noise from the silences and perform noise reduction on the waveform
    silence_times = [[start_times[i], end_times[i]] for i in breaks]
    silence_times = (np.array(silence_times) * hparams.sample_rate).astype(np.int)
    noisy_wav = np.concatenate([wav[stime[0]:stime[1]] for stime in silence_times])
    if len(noisy_wav) > hparams.sample_rate * 0.02:
        profile = logmmse.profile_noise(noisy_wav, hparams.sample_rate)
        wav = logmmse.denoise(wav, profile, eta=0)

    # Re-attach segments that are too short
    segments = list(zip(breaks[:-1], breaks[1:]))
    segment_durations = [start_times[end] - end_times[start] for start, end in segments]
    i = 0
    while i < len(segments) and len(segments) > 1:
        if segment_durations[i] < hparams.utterance_min_duration:
            # See if the segment can be re-attached with the right or the left segment
            left_duration = float("inf") if i == 0 else segment_durations[i - 1]
            right_duration = float("inf") if i == len(segments) - 1 else segment_durations[i + 1]
            joined_duration = segment_durations[i] + min(left_duration, right_duration)

            # Do not re-attach if it causes the joined utterance to be too long
            if joined_duration > hparams.hop_size * hparams.max_mel_frames / hparams.sample_rate:
                i += 1
                continue

            # Re-attach the segment with the neighbour of shortest duration
            j = i - 1 if left_duration <= right_duration else i
            segments[j] = (segments[j][0], segments[j + 1][1])
            segment_durations[j] = joined_duration
            del segments[j + 1], segment_durations[j + 1]
        else:
            i += 1

    # Split the utterance
    segment_times = [[end_times[start], start_times[end]] for start, end in segments]
    segment_times = (np.array(segment_times) * hparams.sample_rate).astype(np.int)
    wavs = [wav[segment_time[0]:segment_time[1]] for segment_time in segment_times]
    texts = [" ".join(words[start + 1:end]).replace("  ", " ") for start, end in segments]

    # # DEBUG: play the audio segments (run with -n=1)
    # import sounddevice as sd
    # if len(wavs) > 1:
    #     print("This sentence was split in %d segments:" % len(wavs))
    # else:
    #     print("There are no silences long enough for this sentence to be split:")
    # for wav, text in zip(wavs, texts):
    #     # Pad the waveform with 1 second of silence because sounddevice tends to cut them early
    #     # when playing them. You shouldn't need to do that in your parsers.
    #     wav = np.concatenate((wav, [0] * 16000))
    #     print("\t%s" % text)
    #     sd.play(wav, 16000, blocking=True)
    # print("")

    return wavs, texts


def process_utterance(raw_wav: np.ndarray, text: str, out_dir: Path, basename: str,
                      skip_existing: bool, hparams, trim_silence=True):
    ## FOR REFERENCE:
    # For you not to lose your head if you ever wish to change things here or implement your own
    # synthesizer.
    # - Both the audios and the mel spectrograms are saved as numpy arrays
    # - There is no processing done to the audios that will be saved to disk beyond volume
    #   normalization (in split_on_silences)
    # - However, pre-emphasis is applied to the audios before computing the mel spectrogram. This
    #   is why we re-apply it on the audio on the side of the vocoder.
    # - Librosa pads the waveform before computing the mel spectrogram. Here, the waveform is saved
    #   without extra padding. This means that you won't have an exact relation between the length
    #   of the wav and of the mel spectrogram. See the vocoder data loader.


    # Skip existing utterances if needed
    mel_fpath = out_dir.joinpath("mels", "mel-%s.npy" % basename)
    mfcc_fpath = out_dir.joinpath("mfccs", "mfcc-%s.npy" % basename)
    wav_fpath = out_dir.joinpath("audio", "audio-%s.npy" % basename)
    if skip_existing and mel_fpath.exists() and mfcc_fpath.exists() and wav_fpath.exists():
        return None
    
    # Trim silence
    wav = speaker_encoder.preprocess_wav(raw_wav, normalize=False, trim_silence=trim_silence)

    # Skip utterances that are too short
    if len(wav) < hparams.utterance_min_duration * hparams.sample_rate:
        return None

    # Compute the mel spectrogram
    mel_spectrogram = audio.melspectrogram(wav, hparams).astype(np.float32)
    mel_frames = mel_spectrogram.shape[1]

    # Skip utterances that are too long
    if mel_frames > hparams.max_mel_frames and hparams.clip_mels_length:
        return None
    
    # add MFCC
    mfcc = get_mfcc(raw_wav, hparams.sample_rate, mean_signal_length=130000)

    # Write the spectrogram, embed and audio to disk
    np.save(mel_fpath, mel_spectrogram.T, allow_pickle=False)
    np.save(wav_fpath, wav, allow_pickle=False)
    np.save(mfcc_fpath, mfcc, allow_pickle=False)

    # Return a tuple describing this training example
    return wav_fpath.name, mel_fpath.name, "speaker-embed-%s.npy" % basename, len(wav), mel_frames, text, mfcc_fpath.name, "emotion-embed-%s.npy" % basename


def embed_utterance(fpath_batch, encoder_model_fpaths, model):
    speaker_encoder_fpath, emotion_encoder_fpath = encoder_model_fpaths
    if not speaker_encoder.is_loaded():
        speaker_encoder.load_model(speaker_encoder_fpath)
    wavs = [speaker_encoder.preprocess_wav(np.load(fpaths[0])) for fpaths in fpath_batch]
    speaker_embed_fpaths = [fpaths[1] for fpaths in fpath_batch]
    mfccs = [np.load(fpaths[2]) for fpaths in fpath_batch]
    emotion_embed_fpaths = [fpaths[3] for fpaths in fpath_batch]

    #### create speaker embeddings ####
    # Compute the speaker embedding of the utterance
    for i, wav in enumerate(wavs):
        embed = speaker_encoder.embed_utterance(wav)
        np.save(speaker_embed_fpaths[i], embed, allow_pickle=False)

    #### create emotion embeddings ####
    x_source = np.array(mfccs)
    x_feats = model.infer(x_source, model_dir=emotion_encoder_fpath)

    for i, emotion_embed_fpath in enumerate(emotion_embed_fpaths):
        np.save(emotion_embed_fpath, x_feats[i], allow_pickle=False)


def create_embeddings(synthesizer_root: Path, speaker_encoder_model_fpath: Path, emotion_encoder_model_fpath:Path, batch_size: int):
    """
    @author: Jiaxin Ye
    @contact: jiaxin-ye@foxmail.com
    """
    # -*- coding:UTF-8 -*-
    import numpy as np
    import os
    import tensorflow as tf
    from emotion_encoder.Model import TIMNET_Model
    import argparse
    import json

    json_fpath = os.path.join(emotion_encoder_model_fpath, "params.json")
    f = open(json_fpath)
    args = argparse.Namespace(**json.load(f))
        
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth=True 
    session = tf.compat.v1.Session(config=config)
    print(f"###gpus:{gpus}")

    CLASS_LABELS = ("angry", "happy", "neutral", "sad")

    model = TIMNET_Model(args=args, input_shape=(254, 39), class_label=CLASS_LABELS)

    model.create_model()
    
    # create train embeddings
    train_wav_dir = synthesizer_root.joinpath("train/audio")
    train_mfcc_dir = synthesizer_root.joinpath("train/mfccs")
    train_metadata_fpath = synthesizer_root.joinpath("train/train.txt")
    train_speaker_embed_dir = synthesizer_root.joinpath("train/speaker_embeds")
    train_speaker_embed_dir.mkdir(exist_ok=True)
    train_emotion_embed_dir = synthesizer_root.joinpath("train/emotion_embeds")
    train_emotion_embed_dir.mkdir(exist_ok=True)
    assert train_wav_dir.exists() and train_mfcc_dir.exists() and train_metadata_fpath.exists() and train_speaker_embed_dir.exists() and train_emotion_embed_dir.exists()

    # Gather the input wave filepath and the target output embed filepath
    with train_metadata_fpath.open("r") as metadata_file:
        metadata = [line.split("|") for line in metadata_file]
        metadata_len = len(metadata)
        iters=metadata_len//batch_size
        residual_size=metadata_len-iters*batch_size
        fpath_batches=[]
        for iter in range(iters):
            fpath_batch=[]
            for i in range(batch_size):
                m = metadata[iter*batch_size+i]
                fpath_batch.append((train_wav_dir.joinpath(m[0].strip()), train_speaker_embed_dir.joinpath(m[2].strip()), train_mfcc_dir.joinpath(m[6].strip()), train_emotion_embed_dir.joinpath(m[7].strip())))
            fpath_batches.append(fpath_batch)
        fpath_batch=[]
        for i in range(residual_size):
            m = metadata[iters*batch_size+i]
            fpath_batch.append((train_wav_dir.joinpath(m[0].strip()), train_speaker_embed_dir.joinpath(m[2].strip()), train_mfcc_dir.joinpath(m[6].strip()), train_emotion_embed_dir.joinpath(m[7].strip())))
        fpath_batches.append(fpath_batch)

    for fpath_batch in tqdm(fpath_batches, desc="Embedding", unit="utterance_batches"):
        embed_utterance(fpath_batch, encoder_model_fpaths=[speaker_encoder_model_fpath, emotion_encoder_model_fpath], model=model)
    
    # create dev embeddings
    dev_wav_dir = synthesizer_root.joinpath("dev/audio")
    dev_mfcc_dir = synthesizer_root.joinpath("dev/mfccs")
    dev_metadata_fpath = synthesizer_root.joinpath("dev/dev.txt")
    dev_speaker_embed_dir = synthesizer_root.joinpath("dev/speaker_embeds")
    dev_speaker_embed_dir.mkdir(exist_ok=True)
    dev_emotion_embed_dir = synthesizer_root.joinpath("dev/emotion_embeds")
    dev_emotion_embed_dir.mkdir(exist_ok=True)
    assert dev_wav_dir.exists() and dev_mfcc_dir.exists() and dev_metadata_fpath.exists() and dev_speaker_embed_dir.exists() and dev_emotion_embed_dir.exists()

    # Gather the input wave filepath and the target output embed filepath
    with dev_metadata_fpath.open("r") as metadata_file:
        metadata = [line.split("|") for line in metadata_file]
        metadata_len = len(metadata)
        iters=metadata_len//batch_size
        residual_size=metadata_len-iters*batch_size
        fpath_batches=[]
        for iter in range(iters):
            fpath_batch=[]
            for i in range(batch_size):
                m = metadata[iter*batch_size+i]
                fpath_batch.append((dev_wav_dir.joinpath(m[0].strip()), dev_speaker_embed_dir.joinpath(m[2].strip()), dev_mfcc_dir.joinpath(m[6].strip()), dev_emotion_embed_dir.joinpath(m[7].strip())))
            fpath_batches.append(fpath_batch)
        fpath_batch=[]
        for i in range(residual_size):
            m = metadata[iters*batch_size+i]
            fpath_batch.append((dev_wav_dir.joinpath(m[0].strip()), dev_speaker_embed_dir.joinpath(m[2].strip()), dev_mfcc_dir.joinpath(m[6].strip()), dev_emotion_embed_dir.joinpath(m[7].strip())))
        fpath_batches.append(fpath_batch)

    for fpath_batch in tqdm(fpath_batches, desc="Embedding", unit="utterance_batches"):
        embed_utterance(fpath_batch, encoder_model_fpaths=[speaker_encoder_model_fpath, emotion_encoder_model_fpath], model=model)