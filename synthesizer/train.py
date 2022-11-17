from datetime import datetime
from functools import partial
from pathlib import Path
from os.path import exists

import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader

from synthesizer import audio
from synthesizer.models.tacotron import Tacotron
from synthesizer.synthesizer_dataset import SynthesizerDataset, collate_synthesizer
from synthesizer.utils import ValueWindow, data_parallel_workaround
from synthesizer.utils.plot import plot_spectrogram
from synthesizer.utils.symbols import symbols
from synthesizer.utils.text import sequence_to_text
from vocoder.display import *
from utils.profiler import Profiler


def np_now(x: torch.Tensor): return x.detach().cpu().numpy()


def time_string():
    return datetime.now().strftime("%Y-%m-%d %H:%M")


def sync(device: torch.device):
    # For correct profiling (cuda operations are async)
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def train(run_id: str, syn_dir: Path, models_dir: Path, save_every: int,  backup_every: int, force_restart: bool,
          hparams):
    models_dir.mkdir(exist_ok=True)

    model_dir = models_dir.joinpath(run_id)
    plot_dir = model_dir.joinpath("plots")
    wav_dir = model_dir.joinpath("wavs")
    mel_output_dir = model_dir.joinpath("mel-spectrograms")
    meta_folder = model_dir.joinpath("metas")
    model_dir.mkdir(exist_ok=True)
    plot_dir.mkdir(exist_ok=True)
    wav_dir.mkdir(exist_ok=True)
    mel_output_dir.mkdir(exist_ok=True)
    meta_folder.mkdir(exist_ok=True)

    weights_fpath = model_dir / f"synthesizer.pt"
    train_metadata_fpath = syn_dir.joinpath("train-clean/train.txt")
    dev_metadata_fpath = syn_dir.joinpath("dev-clean/dev.txt")

    print("Checkpoint path: {}".format(weights_fpath))
    print("Loading training data from: {}".format(train_metadata_fpath))
    print("Using model: Tacotron")

    # Bookkeeping
    time_window = ValueWindow(100)
    loss_window = ValueWindow(100)

    # From WaveRNN/train_tacotron.py
    if torch.cuda.is_available():
        device = torch.device("cuda")

        for session in hparams.tts_schedule:
            _, _, _, batch_size = session
            if batch_size % torch.cuda.device_count() != 0:
                raise ValueError("`batch_size` must be evenly divisible by n_gpus!")
    else:
        device = torch.device("cpu")
    print("Using device:", device)

    # Instantiate Tacotron Model
    print("\nInitialising Tacotron Model...\n")
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
                     dropout=hparams.tts_dropout,
                     stop_threshold=hparams.tts_stop_threshold,
                     speaker_embedding_size=hparams.speaker_embedding_size).to(device)

    # Initialize the optimizer
    optimizer = optim.Adam(model.parameters())
    
    train_loss_file_path = "synthesizer_loss/synthesizer_train_loss.npy"
    dev_loss_file_path = "synthesizer_loss/synthesizer_dev_loss.npy"
    
    # Load the weights
    if force_restart or not weights_fpath.exists():
        print("\nStarting the training of Tacotron from scratch\n")
        model.save(weights_fpath)

        # Embeddings metadata
        char_embedding_fpath = meta_folder.joinpath("CharacterEmbeddings.tsv")
        with open(char_embedding_fpath, "w", encoding="utf-8") as f:
            for symbol in symbols:
                if symbol == " ":
                    symbol = "\\s"  # For visual purposes, swap space with \s

                f.write("{}\n".format(symbol))
                
        losses = []
        dev_losses = []

    else:
        print("\nLoading weights at %s" % weights_fpath)
        model.load(weights_fpath, optimizer)
        print("Tacotron weights loaded from step %d" % model.step)
        losses = list(np.load(train_loss_file_path)) if exists(train_loss_file_path) else []
        dev_losses = list(np.load(dev_loss_file_path)) if exists(dev_loss_file_path) else []
        
    # Initialize the dataset
    train_mel_dir = syn_dir.joinpath("train-clean/mels")
    train_embed_dir = syn_dir.joinpath("train-clean/embeds")
    dev_mel_dir = syn_dir.joinpath("dev-clean/mels")
    dev_embed_dir = syn_dir.joinpath("dev-clean/embeds")
    train_dataset = SynthesizerDataset(train_metadata_fpath, train_mel_dir, train_embed_dir, hparams)
    dev_dataset = SynthesizerDataset(dev_metadata_fpath, dev_mel_dir, dev_embed_dir, hparams)

    best_loss_file_path = "src/synthesizer_loss/best_loss.npy"
    best_loss = np.load(best_loss_file_path)[0] if exists(best_loss_file_path) else 1

    # profiler = Profiler(summarize_every=10, disabled=False)
    for i, session in enumerate(hparams.tts_schedule):
        current_step = model.get_step()

        r, lr, max_step, batch_size = session

        training_steps = max_step - current_step

        # Do we need to change to the next session?
        if current_step >= max_step:
            # Are there no further sessions than the current one?
            if i == len(hparams.tts_schedule) - 1:
                # We have completed training. Save the model and exit
                model.save(weights_fpath, optimizer)
                break
            else:
                # There is a following session, go to it
                continue

        model.r = r

        # Begin the training
        simple_table([(f"Steps with r={r}", str(training_steps // 1000) + "k Steps"),
                      ("Batch Size", batch_size),
                      ("Learning Rate", lr),
                      ("Outputs/Step (r)", model.r)])

        for p in optimizer.param_groups:
            p["lr"] = lr

        collate_fn = partial(collate_synthesizer, r=r, hparams=hparams)
        train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn, pin_memory=True)
        
        total_iters = len(train_dataset)
        steps_per_epoch = np.ceil(total_iters / batch_size).astype(np.int32)
        epochs = np.ceil(training_steps / steps_per_epoch).astype(np.int32)

        for epoch in range(1, epochs+1):
            for i, (texts, mels, embeds, idx) in enumerate(train_dataloader, 1):
                start_time = time.time()

                # profiler.tick("Blocking, waiting for batch (threaded)")

                # Generate stop tokens for training
                stop = torch.ones(mels.shape[0], mels.shape[2])
                for j, k in enumerate(idx):
                    stop[j, :int(train_dataset.metadata[k][4])-1] = 0

                texts = texts.to(device)
                mels = mels.to(device)
                embeds = embeds.to(device)
                stop = stop.to(device)

                # sync(device)
                # profiler.tick("Data to %s" % device)

                # Forward pass
                # Parallelize model onto GPUS using workaround due to python bug
                # if device.type == "cuda" and torch.cuda.device_count() > 1:
                #     m1_hat, m2_hat, attention, stop_pred = data_parallel_workaround(model, texts, mels, embeds)
                # else:
                m1_hat, m2_hat, attention, stop_pred = model(texts, mels, embeds)
                # sync(device)
                # profiler.tick("Forward pass")

                # Backward pass
                m1_loss = F.mse_loss(m1_hat, mels) + F.l1_loss(m1_hat, mels)
                m2_loss = F.mse_loss(m2_hat, mels)
                stop_loss = F.binary_cross_entropy(stop_pred, stop)

                loss = m1_loss + m2_loss + stop_loss

                # sync(device)
                # profiler.tick("Loss")

                optimizer.zero_grad()
                loss.backward()

                # profiler.tick("Backward pass")

                if hparams.tts_clip_grad_norm is not None:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), hparams.tts_clip_grad_norm)
                    if np.isnan(grad_norm.cpu()):
                        print("grad_norm was NaN!")

                optimizer.step()
                # profiler.tick("Parameter update")

                time_window.append(time.time() - start_time)
                loss_window.append(loss.item())

                step = model.get_step()
                k = step // 1000

                msg = f"| Epoch: {epoch}/{epochs} ({i}/{steps_per_epoch}) | Train Loss: {loss_window.average:#.4} | " \
                      f"{1./time_window.average:#.2} steps/s | Step: {k}k | "
                stream(msg)

                # Backup or save model as appropriate
                # if backup_every != 0 and step % backup_every == 0 :
                #     backup_fpath = weights_fpath.parent / f"synthesizer_{k:06d}.pt"
                #     model.save(backup_fpath, optimizer)

                if save_every != 0 and i % save_every == 0:
                    dev_loss = validate(dev_dataset, model, collate_fn)
                    msg = f"\n| Epoch: {epoch}/{epochs} ({i}/{steps_per_epoch}) | Train Loss: {loss_window.average:#.4} | " \
                          f"Dev Loss: {dev_loss:#.4} | {1./time_window.average:#.2} steps/s | Step: {k}k | "
                    print(msg)
                    losses.append(loss_window.average)
                    np.save(train_loss_file_path, np.array(losses, dtype=float))

                    dev_losses.append(dev_loss)
                    np.save(dev_loss_file_path, np.array(dev_losses, dtype=float))

                    if dev_loss < best_loss:
                        # Must save latest optimizer state to ensure that resuming training
                        # doesn't produce artifacts
                        best_loss = dev_loss
                        np.save(best_loss_file_path, np.array([best_loss]))
                        model.save(weights_fpath, optimizer)

                # Evaluate model to generate dev samples
                # epoch_eval = hparams.tts_eval_interval == -1 and i == steps_per_epoch  # If epoch is done
                # step_eval = hparams.tts_eval_interval > 0 and i % hparams.tts_eval_interval == 0  # Every N steps
                # if step_eval:
                    # generate train samples
                    # for sample_idx in range(hparams.tts_eval_num_samples):
                    #     # At most, generate samples equal to number in the batch
                    #     if sample_idx + 1 <= len(texts):
                    #         # Remove padding from mels using frame length in metadata
                    #         mel_length = int(train_dataset.metadata[idx[sample_idx]][4])
                    #         mel_prediction = np_now(m2_hat[sample_idx]).T[:mel_length]
                    #         target_spectrogram = np_now(mels[sample_idx]).T[:mel_length]
                    #         attention_len = mel_length // model.r

                    #         eval_model(attention=np_now(attention[sample_idx][:, :attention_len]),
                    #                    mel_prediction=mel_prediction,
                    #                    target_spectrogram=target_spectrogram,
                    #                    input_seq=np_now(texts[sample_idx]),
                    #                    step=step,
                    #                    plot_dir=plot_dir,
                    #                    mel_output_dir=mel_output_dir,
                    #                    wav_dir=wav_dir,
                    #                    sample_num=sample_idx + 1,
                    #                    loss=loss,
                    #                    hparams=hparams,
                    #                    if_dev="train")

                    # generate dev samples
                    # for sample_idx in range(hparams.tts_eval_num_samples):
                    #     # At most, generate samples equal to number in the batch
                    #     if sample_idx + 1 <= len(dev_input_texts):
                    #         # Remove padding from mels using frame length in metadata
                    #         mel_length = int(dev_dataset.metadata[dev_idx[sample_idx]][4])
                    #         dev_mel_prediction = np_now(dev_m2_hat[sample_idx]).T[:mel_length]
                    #         target_spectrogram = np_now(dev_target_mels[sample_idx]).T[:mel_length]
                    #         attention_len = mel_length // model.r

                    #         eval_model(attention=np_now(dev_attention[sample_idx][:, :attention_len]),
                    #                    mel_prediction=dev_mel_prediction,
                    #                    target_spectrogram=target_spectrogram,
                    #                    input_seq=np_now(dev_input_texts[sample_idx]),
                    #                    step=step,
                    #                    plot_dir=plot_dir,
                    #                    mel_output_dir=mel_output_dir,
                    #                    wav_dir=wav_dir,
                    #                    sample_num=sample_idx + 1,
                    #                    loss=dev_loss,
                    #                    hparams=hparams,
                    #                    if_dev="dev")

                # Break out of loop to update training schedule
                if step >= max_step:
                    break

            # Add line break after every epoch
            print("")


def eval_model(attention, mel_prediction, target_spectrogram, input_seq, step,
               plot_dir, mel_output_dir, wav_dir, sample_num, loss, hparams, if_dev = None):
    # Save some results for evaluation
    attention_path = str(plot_dir.joinpath("{}_attention_step_{}_sample_{}".format(if_dev, step, sample_num)))
    save_attention(attention, attention_path)

    # save predicted mel spectrogram to disk (debug)
    mel_output_fpath = mel_output_dir.joinpath("{}-mel-prediction-step-{}_sample_{}.npy".format(if_dev, step, sample_num))
    np.save(str(mel_output_fpath), mel_prediction, allow_pickle=False)

    # save griffin lim inverted wav for debug (mel -> wav)
    wav = audio.inv_mel_spectrogram(mel_prediction.T, hparams)
    wav_fpath = wav_dir.joinpath("{}-step-{}-wave-from-mel_sample_{}.wav".format(if_dev, step, sample_num))
    audio.save_wav(wav, str(wav_fpath), sr=hparams.sample_rate)

    # save real and predicted mel-spectrogram plot to disk (control purposes)
    spec_fpath = plot_dir.joinpath("{}-step-{}-mel-spectrogram_sample_{}.png".format(if_dev, step, sample_num))
    title_str = "{}, {}, step={}, {} loss={:.5f}".format("Tacotron", time_string(), step, if_dev, loss)
    plot_spectrogram(mel_prediction, str(spec_fpath), title=title_str,
                     target_spectrogram=target_spectrogram,
                     max_len=target_spectrogram.size // hparams.num_mels)
    print("Input at step {}: {}".format(step, sequence_to_text(input_seq)))


def validate(dataset, model, collate_fn):
    model.eval()
    with torch.no_grad():
        losses = []
        dataloader = DataLoader(dataset, 32, num_workers=4, shuffle=False, collate_fn=collate_fn)
        for i, (texts, mels, embeds, idx) in enumerate(dataloader, 1):
            # Generate stop tokens for training
            stop = torch.ones(mels.shape[0], mels.shape[2])
            for j, k in enumerate(idx):
                stop[j, :int(dataset.metadata[k][4])-1] = 0

            texts = texts.cuda()
            mels = mels.cuda()
            embeds = embeds.cuda()
            stop = stop.cuda()

            # Forward pass
            # Parallelize model onto GPUS using workaround due to python bug
            # if device.type == "cuda" and torch.cuda.device_count() > 1:
            #     m1_hat, m2_hat, attention, stop_pred = data_parallel_workaround(model, texts, mels, embeds)
            # else:
            m1_hat, m2_hat, attention, stop_pred = model(texts, mels, embeds)

            # Backward pass
            m1_loss = F.mse_loss(m1_hat, mels) + F.l1_loss(m1_hat, mels)
            m2_loss = F.mse_loss(m2_hat, mels)
            stop_loss = F.binary_cross_entropy(stop_pred, stop)

            loss = m1_loss + m2_loss + stop_loss
            losses.append(loss.item())
    model.train()
    torch.cuda.empty_cache()
    return sum(losses) / len(losses)
