import platform
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from synthesizer.hparams import hparams_debug_string
from synthesizer.models.tacotron import Tacotron
from synthesizer.synthesizer_dataset import SynthesizerDataset, collate_synthesizer
from synthesizer.utils import data_parallel_workaround
from synthesizer.utils.symbols import symbols


def run_synthesis(in_dir: Path, out_dir: Path, syn_model_fpath: Path, hparams):
    # This generates ground truth-aligned mels for vocoder training
    train_in_dir = in_dir.joinpath("train-clean")
    train_out_dir = out_dir.joinpath("train-clean")
    dev_in_dir = in_dir.joinpath("dev-clean")
    dev_out_dir = out_dir.joinpath("dev-clean")
    train_synth_dir = train_out_dir / "mels_gta"
    train_synth_dir.mkdir(exist_ok=True, parents=True)
    dev_synth_dir = dev_out_dir / "mels_gta"
    dev_synth_dir.mkdir(exist_ok=True, parents=True)
    print(hparams_debug_string())

    # Check for GPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
        if hparams.synthesis_batch_size % torch.cuda.device_count() != 0:
            raise ValueError("`hparams.synthesis_batch_size` must be evenly divisible by n_gpus!")
    else:
        device = torch.device("cpu")
    print("Synthesizer using device:", device)

    # Instantiate Tacotron model
    model = Tacotron(embed_dims=hparams.tts_embed_dims,
                     num_chars=len(symbols),
                     encoder_dims=hparams.tts_encoder_dims,
                     decoder_dims=hparams.tts_decoder_dims,
                     n_mels=hparams.num_mels,
                     fft_bins=hparams.num_mels,
                     postnet_dims=hparams.tts_postnet_dims,
                     encoder_K=hparams.tts_encoder_K,
                     lstm_dims=hparams.tts_lstm_dims,
                     postnet_K=hparams.tts_postnet_K,
                     num_highways=hparams.tts_num_highways,
                     dropout=0., # Use zero dropout for gta mels
                     stop_threshold=hparams.tts_stop_threshold,
                     speaker_embedding_size=hparams.speaker_embedding_size).to(device)

    # Load the weights
    print("\nLoading weights at %s" % syn_model_fpath)
    model.load(syn_model_fpath)
    print("Tacotron weights loaded from step %d" % model.step)

    # Synthesize using same reduction factor as the model is currently trained
    r = np.int32(model.r)

    # Set model to eval mode (disable gradient and zoneout)
    model.eval()

    # Initialize the dataset
    train_metadata_fpath = train_in_dir.joinpath("train.txt")
    train_mel_dir = train_in_dir.joinpath("mels")
    train_embed_dir = train_in_dir.joinpath("embeds")
    dev_metadata_fpath = dev_in_dir.joinpath("dev.txt")
    dev_mel_dir = dev_in_dir.joinpath("mels")
    dev_embed_dir = dev_in_dir.joinpath("embeds")

    train_dataset = SynthesizerDataset(train_metadata_fpath, train_mel_dir, train_embed_dir, hparams)
    dev_dataset = SynthesizerDataset(dev_metadata_fpath, dev_mel_dir, dev_embed_dir, hparams)
    collate_fn = partial(collate_synthesizer, r=r, hparams=hparams)
    train_data_loader = DataLoader(train_dataset, hparams.synthesis_batch_size, collate_fn=collate_fn, num_workers=2)
    dev_data_loader = DataLoader(dev_dataset, hparams.synthesis_batch_size, collate_fn=collate_fn, num_workers=2)

    # Generate train GTA mels
    train_meta_out_fpath = train_out_dir / "synthesized.txt"
    with train_meta_out_fpath.open("w") as file:
        for i, (texts, mels, embeds, idx) in tqdm(enumerate(train_data_loader), total=len(train_data_loader)):
            texts, mels, embeds = texts.to(device), mels.to(device), embeds.to(device)

            # Parallelize model onto GPUS using workaround due to python bug
            # if device.type == "cuda" and torch.cuda.device_count() > 1:
            #     _, mels_out, _ = data_parallel_workaround(model, texts, mels, embeds)
            # else:
            _, mels_out, _, _ = model(texts, mels, embeds)

            for j, k in enumerate(idx):
                # Note: outputs mel-spectrogram files and target ones have same names, just different folders
                mel_filename = Path(train_synth_dir).joinpath(train_dataset.metadata[k][1])
                mel_out = mels_out[j].detach().cpu().numpy().T

                # Use the length of the ground truth mel to remove padding from the generated mels
                mel_out = mel_out[:int(train_dataset.metadata[k][4])]

                # Write the spectrogram to disk
                np.save(mel_filename, mel_out, allow_pickle=False)

                # Write metadata into the synthesized file
                file.write("|".join(train_dataset.metadata[k]))
                
    # Generate dev GTA mels
    dev_meta_out_fpath = dev_out_dir / "synthesized.txt"
    with dev_meta_out_fpath.open("w") as file:
        for i, (texts, mels, embeds, idx) in tqdm(enumerate(dev_data_loader), total=len(dev_data_loader)):
            texts, mels, embeds = texts.to(device), mels.to(device), embeds.to(device)

            # Parallelize model onto GPUS using workaround due to python bug
            # if device.type == "cuda" and torch.cuda.device_count() > 1:
            #     _, mels_out, _ = data_parallel_workaround(model, texts, mels, embeds)
            # else:
            _, mels_out, _, _ = model(texts, mels, embeds)

            for j, k in enumerate(idx):
                # Note: outputs mel-spectrogram files and target ones have same names, just different folders
                mel_filename = Path(dev_synth_dir).joinpath(dev_dataset.metadata[k][1])
                mel_out = mels_out[j].detach().cpu().numpy().T

                # Use the length of the ground truth mel to remove padding from the generated mels
                mel_out = mel_out[:int(dev_dataset.metadata[k][4])]

                # Write the spectrogram to disk
                np.save(mel_filename, mel_out, allow_pickle=False)

                # Write metadata into the synthesized file
                file.write("|".join(dev_dataset.metadata[k]))

