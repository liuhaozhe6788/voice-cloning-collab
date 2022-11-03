from pathlib import Path

import torch

from encoder.data_objects import DataLoader, Train_Dataset, Dev_Dataset
from encoder.model import SpeakerEncoder
from encoder.params_model import *


def test(run_id: str, clean_data_root: Path, models_dir: Path):
    test_dataset = Dev_Dataset(clean_data_root.joinpath("voxceleb-test"))
    num_speakers = len(test_dataset)
    test_loader = DataLoader(
        test_dataset,
        num_speakers,
        200,
        num_workers=4,
        pin_memory=True
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  ####modified####

    # Create the model and the optimizer
    model = SpeakerEncoder(device, loss_device)
    
    # Configure file path for the model
    model_dir = models_dir / run_id
    model_dir.mkdir(exist_ok=True, parents=True)
    state_fpath = model_dir / "encoder.pt"

    # Load any existing model
    if state_fpath.exists():
        print("Found existing model \"%s\", loading it and test." % run_id)
        checkpoint = torch.load(state_fpath)
        model.load_state_dict(checkpoint["model_state"])
        
    model.eval()
    with torch.no_grad():
        for step, speaker_batch in enumerate(test_loader, 1):
            frames = torch.from_numpy(speaker_batch.data).to(device)
            embeds = model.forward(frames)
            embeds_loss = embeds.view((num_speakers, utterances_per_speaker, -1)).to(loss_device)
            _, eer = model.loss(embeds_loss)
            return eer
        
        
if __name__ == "__main__":
    run_id = "default"
    clean_data_root = Path("C:\\liuhaozhe\\voice-cloning\\data\\SV2TTS\\encoder")
    model_dir = Path("C:\\liuhaozhe\\voice-cloning\\src\\saved_models")
    for i in range(10):
        print(f"test eer = {test(run_id, clean_data_root, model_dir)}")
    