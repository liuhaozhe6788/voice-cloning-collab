import argparse
from ctypes import alignment
import os
from pathlib import Path
import spacy
import time 

import PyQt5

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
    parser.add_argument("--emotion_encoder_model_fpath", type=Path,
                        default="INTERSECT_46_dilation_8_dropout_05_add_esd_npairLoss", help=\
        "Path your trained emotion encoder model.")
    parser.add_argument("--weight", type=float, default=1,
                        help="weight of input audio for voice filter")
    parser.add_argument("--griffin_lim",
                        action="store_true",
                        help="if True, use griffin-lim, else use vocoder")
    parser.add_argument("--cpu", action="store_false", help=\
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

    import torch
    import speaker_encoder.inference
    import speaker_encoder.params_data 
    from synthesizer.inference import Synthesizer
    from vocoder import inference as vocoder
    import json
    import tensorflow as tf
    from emotion_encoder.Model import TIMNET_Model

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

    if not args.griffin_lim:
        print("Preparing the encoder, the synthesizer and the vocoder...")
    else:
        print("Preparing the encoder and the synthesizer...")

    speaker_encoder.inference.load_model(list(args.models_dir.glob(f"{args.run_id}/encoder.pt"))[0])
    synthesizer = Synthesizer(list(args.models_dir.glob(f"{args.run_id}/synthesizer.pt"))[0], model_name="EmotionTacotron")
    if not args.griffin_lim:
        vocoder.load_model(list(args.models_dir.glob(f"{args.run_id}/vocoder.pt"))[0])

    json_fpath = args.models_dir / args.run_id / args.emotion_encoder_model_fpath / 'params.json'
    f = open(json_fpath)
    emotion_encoder_args = argparse.Namespace(**json.load(f))
    f.close()

    os.environ['CUDA_VISIBLE_DEVICES'] = emotion_encoder_args.gpu
    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth=True 
    session = tf.compat.v1.Session(config=config)
    print(f"###gpus:{gpus}")

    CLASS_LABELS = ("angry", "happy", "neutral", "sad", "surprise")
    emotion_encoder = TIMNET_Model(args=emotion_encoder_args, input_shape=(626, 39), class_label=CLASS_LABELS)
    emotion_encoder.create_model()

    ## Interactive speech generation
    print("This is a GUI-less example of interface to SV2TTS. The purpose of this script is to "
          "show how you can interface this project easily with your own. See the source code for "
          "an explanation of what is happening.\n")

    print("Interactive generation loop")
    num_generated = 0

    nlp = spacy.load('en_core_web_sm')
    weight = arg_dict["weight"] # 声音美颜的用户语音权重
    amp = 1

    num_of_input_audio = 1

    """TODO:
    1.输入音频文件
    2.预处理
    3.获取话者特征向量和情感特征向量
    4.输入需要转换为语音的文本
    5.预处理文本
    6.生成新语音
    """