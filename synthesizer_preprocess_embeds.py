from synthesizer.preprocess import create_embeddings
from utils.argutils import print_args
from pathlib import Path
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Creates embeddings for the synthesizer from the LibriSpeech utterances.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("synthesizer_root", type=Path, help=\
        "Path to the synthesizer training data that contains the audios and the train.txt file. "
        "If you let everything as default, it should be <datasets_root>/SV2TTS/synthesizer/.")
    parser.add_argument("--speaker_encoder_model_fpath", type=Path,
                        default="saved_models/default/encoder.pt", help=\
        "Path your trained speaker encoder model.")
    parser.add_argument("--emotion_encoder_model_fpath", type=Path,
                        default="saved_models/default/INTERSECT_46_dilation_8_dropout_05", help=\
        "Path your trained emotion encoder model.")
    parser.add_argument("-n", "--n_processes", type=int, default=4, help= \
        "Number of parallel processes. An encoder is created for each, so you may need to lower "
        "this value on GPUs with low memory. Set it to 1 if CUDA is unhappy.")
    args = parser.parse_args()

    # Preprocess the dataset
    print_args(args, parser)
    create_embeddings(**vars(args))
