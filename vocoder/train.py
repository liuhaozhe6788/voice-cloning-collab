import time
from pathlib import Path
from os.path import exists

import numpy as np
import torch
import torch.nn.functional as F
from torch import no_grad, optim
from torch.utils.data import DataLoader

import vocoder.hparams as hp
from vocoder.display import stream, simple_table
from vocoder.distribution import discretized_mix_logistic_loss
from vocoder.gen_wavernn import gen_devset
from vocoder.models.fatchord_version import WaveRNN
from vocoder.vocoder_dataset import VocoderDataset, collate_vocoder
from vocoder.utils import ValueWindow
from utils.profiler import Profiler


def train(run_id: str, syn_dir: Path, voc_dir: Path, models_dir: Path, ground_truth: bool, save_every: int,
          backup_every: int, force_restart: bool):
    # Check to make sure the hop length is correctly factorised
    train_syn_dir = syn_dir.joinpath("train-clean")
    train_voc_dir = voc_dir.joinpath("train-clean")    
    dev_syn_dir = syn_dir.joinpath("dev-clean")
    dev_voc_dir = voc_dir.joinpath("dev-clean")
    assert np.cumprod(hp.voc_upsample_factors)[-1] == hp.hop_length

    # Instantiate the model
    print("Initializing the model...")
    model = WaveRNN(
        rnn_dims=hp.voc_rnn_dims,
        fc_dims=hp.voc_fc_dims,
        bits=hp.bits,
        pad=hp.voc_pad,
        upsample_factors=hp.voc_upsample_factors,
        feat_dims=hp.num_mels,
        compute_dims=hp.voc_compute_dims,
        res_out_dims=hp.voc_res_out_dims,
        res_blocks=hp.voc_res_blocks,
        hop_length=hp.hop_length,
        sample_rate=hp.sample_rate,
        mode=hp.voc_mode
    )

    if torch.cuda.is_available():
        model = model.cuda()

    # Initialize the optimizer
    optimizer = optim.Adam(model.parameters())
    for p in optimizer.param_groups:
        p["lr"] = hp.voc_lr
    loss_func = F.cross_entropy if model.mode == "RAW" else discretized_mix_logistic_loss
    train_loss_window = ValueWindow(100)

    # Load the weights
    model_dir = models_dir / run_id
    model_dir.mkdir(exist_ok=True)
    weights_fpath = model_dir / "vocoder.pt"
    train_loss_file_path = "src/vocoder_loss/vocoder_train_loss.npy"
    dev_loss_file_path = "src/vocoder_loss/vocoder_dev_loss.npy"
    if force_restart or not weights_fpath.exists():
        print("\nStarting the training of WaveRNN from scratch\n")
        model.save(weights_fpath, optimizer)
        losses = []
        dev_losses = []
    else:
        print("\nLoading weights at %s" % weights_fpath)
        model.load(weights_fpath, optimizer)
        print("WaveRNN weights loaded from step %d" % model.step)
        losses = list(np.load(train_loss_file_path)) if exists(train_loss_file_path) else []
        dev_losses = list(np.load(dev_loss_file_path)) if exists(dev_loss_file_path) else []

    # Initialize the dataset
    train_metadata_fpath = train_syn_dir.joinpath("train.txt") if ground_truth else \
        train_voc_dir.joinpath("synthesized.txt")
    train_mel_dir = train_syn_dir.joinpath("mels") if ground_truth else train_voc_dir.joinpath("mels_gta")
    train_wav_dir = train_syn_dir.joinpath("audio")
    train_dataset = VocoderDataset(train_metadata_fpath, train_mel_dir, train_wav_dir)
    
    dev_metadata_fpath = dev_syn_dir.joinpath("dev.txt") if ground_truth else \
        dev_voc_dir.joinpath("synthesized.txt")
    dev_mel_dir = dev_syn_dir.joinpath("mels") if ground_truth else dev_voc_dir.joinpath("mels_gta")
    dev_wav_dir = dev_syn_dir.joinpath("audio")
    dev_dataset = VocoderDataset(dev_metadata_fpath, dev_mel_dir, dev_wav_dir)
    train_dataloader = DataLoader(train_dataset, hp.voc_batch_size, shuffle=True, num_workers=8, collate_fn=collate_vocoder, pin_memory=True)
    dev_dataloader = DataLoader(dev_dataset, hp.voc_batch_size, shuffle=True, num_workers=8, collate_fn=collate_vocoder, pin_memory=True)
    dev_dataloader_ = DataLoader(dev_dataset, 1, shuffle=True)

    # Begin the training
    simple_table([('Batch size', hp.voc_batch_size),
                  ('LR', hp.voc_lr),
                  ('Sequence Len', hp.voc_seq_len)])
    best_loss_file_path = "src/vocoder_loss/best_loss.npy"
    best_loss = np.load(best_loss_file_path)[0] if exists(best_loss_file_path) else 1000

    # profiler = Profiler(summarize_every=10, disabled=False)
    for epoch in range(1, 350):
        start = time.time()

        for i, (x, y, m) in enumerate(train_dataloader, 1):
            model.train()
            # profiler.tick("Blocking, waiting for batch (threaded)")
            if torch.cuda.is_available():
                x, m, y = x.cuda(), m.cuda(), y.cuda()
            # profiler.tick("Data to cuda")

            # Forward pass
            y_hat = model(x, m)
            if model.mode == 'RAW':
                y_hat = y_hat.transpose(1, 2).unsqueeze(-1)
            elif model.mode == 'MOL':
                y = y.float()
            y = y.unsqueeze(-1)
            # profiler.tick("Forward pass")

            # Backward pass
            loss = loss_func(y_hat, y)
            # profiler.tick("Loss")
            optimizer.zero_grad()
            loss.backward()
            # profiler.tick("Backward pass")
            optimizer.step()
            # profiler.tick("Parameter update")

            speed = i / (time.time() - start)
            train_loss_window.append(loss.item())

            step = model.get_step()
            k = step // 1000

            msg = f"| Epoch: {epoch} ({i}/{len(train_dataloader)}) | " \
                f"Train Loss: {train_loss_window.average:.4f} | " \
                f"{speed:.4f}steps/s | Step: {k}k | "
            stream(msg)

            if backup_every != 0 and i % backup_every == 0 :
                model.checkpoint(model_dir, optimizer)

            if save_every != 0 and i % save_every == 0 :
                dev_loss = validate(dev_dataloader, model, loss_func)
                msg = f"| Epoch: {epoch} ({i}/{len(train_dataloader)}) | " \
                    f"Train Loss: {train_loss_window.average:.4f} | Dev Loss: {dev_loss:.4f} | " \
                    f"{speed:.4f}steps/s | Step: {k}k | "
                stream(msg)
                losses.append(train_loss_window.average)
                np.save(train_loss_file_path, np.array(losses, dtype=float))
                dev_losses.append(dev_loss)
                np.save(dev_loss_file_path, np.array(dev_losses, dtype=float))
                if dev_loss < best_loss :
                    best_loss = dev_loss
                    np.save(best_loss_file_path, np.array([best_loss]))
                    model.save(weights_fpath, optimizer)

            # profiler.tick("Extra saving")

        # gen_devset(model, dev_dataloader_, hp.voc_gen_at_checkpoint, hp.voc_gen_batched,
        #         hp.voc_target, hp.voc_overlap, model_dir)
        print("")

def validate(dataloader, model, loss_func):
    model.eval()
    losses = []
    with no_grad():
        for i, (x, y, m) in enumerate(dataloader, 1):
            if torch.cuda.is_available():
                x, m, y = x.cuda(), m.cuda(), y.cuda()
                y_hat = model(x, m)
                if model.mode == 'RAW':
                    y_hat = y_hat.transpose(1, 2).unsqueeze(-1)
                elif model.mode == 'MOL':
                    y = y.float()
                y = y.unsqueeze(-1)
                loss = loss_func(y_hat, y).item()
                losses.append(loss)
        return sum(losses) / len(losses)