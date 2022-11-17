from pathlib import Path
import numpy as np
from os.path import exists

import torch

from encoder.data_objects import DataLoader, Train_Dataset, Dev_Dataset
from encoder.model import SpeakerEncoder
from encoder.params_model import *
from encoder.visualizations import Visualizations
from utils.profiler import Profiler


def sync(device: torch.device):
    # For correct profiling (cuda operations are async)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def train(run_id: str, clean_data_root: Path, models_dir: Path, umap_every: int, save_every: int,
          backup_every: int, vis_every: int, force_restart: bool, visdom_server: str,
          no_visdom: bool):
    # Create a dataset and a dataloader
    train_dataset = Train_Dataset(clean_data_root.joinpath("train"))
    dev_dataset = Dev_Dataset(clean_data_root.joinpath("dev"))
    train_loader = DataLoader(
        train_dataset,
        speakers_per_batch,
        utterances_per_speaker,
        shuffle=True,
        num_workers=8,
        pin_memory=True
    )
    dev_batch = len(dev_dataset)
    dev_loader = DataLoader(
        dev_dataset,
        dev_batch,
        utterances_per_speaker,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    

    # Setup the device on which to run the forward pass and the loss. These can be different,
    # because the forward pass is faster on the GPU whereas the loss is often (depending on your
    # hyperparameters) faster on the CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # FIXME: currently, the gradient is None if loss_device is cuda
    
    # loss_device = torch.device("cpu")
    loss_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  ####modified####

    # Create the model and the optimizer
    model = SpeakerEncoder(device, loss_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate_init)
    current_lr = learning_rate_init
    init_step = 1

    # Configure file path for the model
    model_dir = models_dir / run_id
    model_dir.mkdir(exist_ok=True, parents=True)
    state_fpath = model_dir / "encoder.pt"

    # Load any existing model
    if not force_restart:
        if state_fpath.exists():
            print("Found existing model \"%s\", loading it and resuming training." % run_id)
            checkpoint = torch.load(state_fpath)
            init_step = checkpoint["step"]
            print(f"Resuming training from step {init_step}")
            model.load_state_dict(checkpoint["model_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            optimizer.param_groups[0]["lr"] = learning_rate_init
        else:
            print("No model \"%s\" found, starting training from scratch." % run_id)
    else:
        print("Starting the training from scratch.")

    # Initialize the visualization environment
    vis = Visualizations(run_id, vis_every, server=visdom_server, disabled=no_visdom)
    vis.log_dataset(train_dataset)
    vis.log_params()
    device_name = str(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
    vis.log_implementation({"Device": device_name})
    
    best_eer_file_path = "encoder_loss/best_eer.npy"
    best_eer = np.load(best_eer_file_path)[0] if exists(best_eer_file_path) else 1

    # Training loop
    profiler = Profiler(summarize_every=1000, disabled=False)
    for step, speaker_batch in enumerate(train_loader, init_step):
        model.train()
        profiler.tick("Blocking, waiting for batch (threaded)")
        # Data to GPU mem
        inputs = torch.from_numpy(speaker_batch.data).to(device)
        sync(device)
        profiler.tick("Data to %s" % device)

        # Forward pass
        embeds = model(inputs)
        sync(device)
        profiler.tick("Forward pass")

        embeds_loss = embeds.view((speakers_per_batch, utterances_per_speaker, -1)).to(loss_device)
        loss, eer = model.loss(embeds_loss)
        sync(loss_device)
        profiler.tick("Loss")

        # Backward pass
        model.zero_grad()  # Sets gradients of all model parameters to zero
        loss.backward()  # Calc gradients of all model parameters
        profiler.tick("Backward pass")
        model.do_gradient_ops()
        optimizer.step()  # do gradient descent of all model parameters
        profiler.tick("Parameter update")

        # Update visualizations
        # learning_rate = optimizer.param_groups[0]["lr"]

        # Overwrite the latest version of the model
        if save_every != 0 and step % save_every == 0:
            current_lr *= 0.995
            update_lr(optimizer, current_lr)
            dev_loss, dev_eer, dev_embeds = validate(dev_loader, model, dev_batch, device, loss_device)
            sync(device)
            sync(loss_device)
            profiler.tick("validate")
            vis.update(loss.item(), eer, step, dev_loss, dev_eer)
            if dev_eer < best_eer:
                best_eer = dev_eer
                np.save(best_eer_file_path, np.array([best_eer]))
                print("Saving the model (step %d)" % step)
                torch.save({
                    "step": step + 1,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                }, state_fpath)
        else:
            vis.update(loss.item(), eer, step)

        # Draw projections and save them to the backup folder
        if umap_every != 0 and step % umap_every == 0:
            print("Drawing and saving projections (step %d)" % step)
            projection_fpath = model_dir / f"umap_{step:06d}.png"
            dev_projection_fpath = model_dir / f"dev_umap_{step:06d}.png"
            embeds = embeds.detach().cpu().numpy()
            dev_embeds = dev_embeds.detach().cpu().numpy()
            vis.draw_projections(embeds, dev_embeds, utterances_per_speaker, step, projection_fpath, dev_projection_fpath)
            vis.save()

        # # Make a backup
        # if backup_every != 0 and step % backup_every == 0:
        #     print("Making a backup (step %d)" % step)
        #     backup_fpath = model_dir / f"encoder_{step:06d}.bak"
        #     torch.save({
        #         "step": step + 1,
        #         "model_state": model.state_dict(),
        #         "optimizer_state": optimizer.state_dict(),
        #     }, backup_fpath)

        profiler.tick("Extras (visualizations, saving)")


def validate(dev_loader: DataLoader, model: SpeakerEncoder, dev_batch, device, loss_device):
    model.eval()
    losses = []
    eers = []
    with torch.no_grad():
        for step, speaker_batch in enumerate(dev_loader, 1):
            frames = torch.from_numpy(speaker_batch.data).to(device)
            embeds = model.forward(frames)
            embeds_loss = embeds.view((dev_batch, utterances_per_speaker, -1)).to(loss_device)
            loss, eer = model.loss(embeds_loss)
            losses.append(loss.item())
            eers.append(eer)
        return sum(losses) / len(losses), sum(eers) / len(eers), embeds.detach()
