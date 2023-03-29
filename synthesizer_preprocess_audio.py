from synthesizer.preprocess import preprocess_librispeech, preprocess_vctk
from synthesizer.hparams import hparams
from utils.argutils import print_args
from pathlib import Path
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocesses audio files from datasets, encodes them as mel spectrograms "
                    "and writes them to  the disk. Audio files are also saved, to be used by the "
                    "vocoder for training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("datasets_root", type=Path, help=\
        "Path to the directory containing your LibriSpeech/TTS datasets.")
    parser.add_argument("-o", "--out_dir", type=Path, default=argparse.SUPPRESS, help=\
        "Path to the output directory that will contain the mel spectrograms, the audios and the "
        "embeds. Defaults to <datasets_root>/SV2TTS/synthesizer/")
    parser.add_argument("-n", "--n_processes", type=int, default=4, help=\
        "Number of processes in parallel.")
    parser.add_argument("-s", "--skip_existing", action="store_true", help=\
        "Whether to overwrite existing files with the same name. Useful if the preprocessing was "
        "interrupted.")
    parser.add_argument("--hparams", type=str, default="", help=\
        "Hyperparameter overrides as a comma-separated list of name-value pairs")
    parser.add_argument("--datasets_names", type=list, default=["LibriSpeech","VCTK"], help=\
        "Name of the dataset directory to process.")
    parser.add_argument("--all_subfolders", type=list, default=["train-clean-100,train-clean-360,dev-clean", "wav48_silence_trimmed"], help=\
        "Comma-separated list of subfolders to process inside your dataset directory")
    args = parser.parse_args()

    # Process the arguments
    if not hasattr(args, "out_dir"):
        args.out_dir = args.datasets_root.joinpath("SV2TTS", "synthesizer")

    # Create directories
    assert args.datasets_root.exists()
    args.out_dir.mkdir(exist_ok=True, parents=True)

    # Preprocess the dataset
    print_args(args, parser)
    args.hparams = hparams.parse(args.hparams)
    preprocess_func = {
        "LibriSpeech": preprocess_librispeech,
        "VCTK": preprocess_vctk,
    }
    args = vars(args)
    for i in range(len(args["datasets_names"])):
        dataset = args["datasets_names"][i]
        subfolders = args["all_subfolders"][i]
        print("Preprocessing %s" % dataset)

        preprocess_func[dataset](datasets_root=args["datasets_root"], out_dir=args["out_dir"], n_processes=args["n_processes"], skip_existing=args["skip_existing"], hparams=args["hparams"],
                       datasets_name=dataset, subfolders=subfolders)
