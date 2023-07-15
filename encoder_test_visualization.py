from pathlib import Path
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE
import umap

import torch

from encoder.data_objects import DataLoader, Train_Dataset, Dev_Dataset
from encoder.model import SpeakerEncoder
from encoder.params_model import *
from encoder.params_data import *

colormap = np.array([
    [76, 255, 0],
    [0, 255, 76],
    [0, 76, 255],
    [0, 127, 70],
    [70, 127, 0],
    [127, 70, 0],
    [255, 0, 0],
    [255, 217, 38],
    [255, 38, 217],
    [38, 217, 255],

    [0, 135, 255],
    [135, 0, 255],
    [255, 135, 0],
    [165, 0, 165],
    [0, 165, 165],
    [165, 165, 0],
    [255, 167, 255],
    [167, 255, 255],
    [255, 255, 167],
    [0, 255, 255],

    [255, 0, 255],
    [255, 255, 0],
    [255, 96, 38],
    [96, 255, 38],
    [38, 96, 255],
    [142, 76, 0],
    [142, 0, 76],
    [0, 76, 142],
    [33, 0, 127],
    [0, 33, 127],

    [33, 127, 0],
    [0, 0, 0],
    [183, 183, 183],
], dtype=np.float) / 255

def draw_scatterplot(x, labels, num_speakers, algo):
    sns.color_palette("tab10")
    colors = [colormap[i] for i in labels]
    plt.scatter(x=x[:, 0], y=x[:, 1], c=colors)
    plt.title(f"{algo}({num_speakers} speakers)")
    if not os.path.exists("dim_reduction_results"):
        os.mkdir("dim_reduction_results")
    plt.savefig(f"dim_reduction_results/{algo}_{num_speakers}.png", dpi=600)  
    plt.clf()

def test_visualization(run_id: str, clean_data_root: Path, models_dir: Path):
    test_dataset = Dev_Dataset(clean_data_root.joinpath("test"))
    num_speakers = len(test_dataset)
    test_loader = DataLoader(
        test_dataset,
        num_speakers,
        utterances_per_speaker,
        shuffle=False,
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
        
            num_speakers_for_visualization = num_speakers
            embeds_cpu = embeds.detach().cpu().numpy()[:num_speakers_for_visualization*utterances_per_speaker, :]
            labels = np.repeat(np.arange(num_speakers_for_visualization), utterances_per_speaker) 

            embeds_pca = PCA(n_components=2).fit_transform(embeds_cpu)
            draw_scatterplot(embeds_pca, labels, num_speakers_for_visualization, "PCA")

            embeds_mds = MDS(n_components=2).fit_transform(embeds_cpu)
            draw_scatterplot(embeds_mds, labels, num_speakers_for_visualization, "MDS")

            embeds_lda = LinearDiscriminantAnalysis(n_components=2).fit_transform(embeds_cpu, labels)
            draw_scatterplot(embeds_lda, labels, num_speakers_for_visualization, "LDA")

            embeds_tsne = TSNE(n_components=2, perplexity=10).fit_transform(embeds_cpu)
            draw_scatterplot(embeds_tsne, labels, num_speakers_for_visualization, "T-SNE")

            embeds_umap = umap.UMAP(n_components=2).fit_transform(embeds_cpu)
            draw_scatterplot(embeds_umap, labels, num_speakers_for_visualization, "UMAP")

            embeds_cpu_zero_op = np.copy(embeds_cpu)
            embeds_cpu_zero_op[embeds_cpu_zero_op < set_zero_thres] = 0

            embeds_tsne = TSNE(n_components=2, perplexity=10).fit_transform(embeds_cpu_zero_op)
            draw_scatterplot(embeds_tsne, labels, num_speakers_for_visualization, "T-SNE_zero_op")

            embeds_umap = umap.UMAP(n_components=2).fit_transform(embeds_cpu_zero_op)
            draw_scatterplot(embeds_umap, labels, num_speakers_for_visualization, "UMAP_zero_op")
        
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Trains the speaker encoder. You must have run encoder_preprocess.py first.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("run_id", type=str, help= \
        "Name for this model. By default, training outputs will be stored to saved_models/<run_id>/. If a model state "
        "from the same run ID was previously saved, the training will restart from there. Pass -f to overwrite saved "
        "states and restart from scratch.")
    parser.add_argument("clean_data_root", type=Path, help= \
        "Path to the output directory of encoder_preprocess.py. If you left the default "
        "output directory when preprocessing, it should be <datasets_root>/SV2TTS/encoder/.")
    parser.add_argument("-m", "--models_dir", type=Path, default="saved_models", help=\
        "Path to the root directory that contains all models. A directory <run_name> will be created under this root."
        "It will contain the saved model weights, as well as backups of those weights and plots generated during "
        "training.")

    args = parser.parse_args()
    args = vars(args)

    test_visualization(**args)
    