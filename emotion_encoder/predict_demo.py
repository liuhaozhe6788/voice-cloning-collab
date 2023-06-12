import numpy as np
import os
import tensorflow as tf
from Model import TIMNET_Model
import argparse
from utils import get_feature

parser = argparse.ArgumentParser()

parser.add_argument('--model_path', type=str, default='./Models/')
parser.add_argument('--result_path', type=str, default='./Results/')
#TODO:修改调用的模型为INTERSECT
parser.add_argument('--test_path', type=str, default='./Test_Models/INTERSECT_46_2023-06-08_22-39-44')
parser.add_argument('--data', type=str, default='INTERSECT')
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--beta1', type=float, default=0.93)
parser.add_argument('--beta2', type=float, default=0.98)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--epoch', type=int, default=60)
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--random_seed', type=int, default=46)
parser.add_argument('--activation', type=str, default='relu')
parser.add_argument('--filter_size', type=int, default=39)
parser.add_argument('--dilation_size', type=int, default=8)# If you want to train model on IEMOCAP, you should modify this parameter to 10 due to the long duration of speech signals.
parser.add_argument('--bidirection', action='store_true')
parser.add_argument('--kernel_size', type=int, default=2)
parser.add_argument('--stack_size', type=int, default=1)
parser.add_argument('--split_fold', type=int, default=10)
parser.add_argument('--gpu', type=str, default='0')

args = parser.parse_args()

if args.data=="IEMOCAP" and args.dilation_size!=10:
    args.dilation_size = 10
elif args.data=="INTERSECT":
    args.dilation_size = 8
    
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True 
session = tf.compat.v1.Session(config=config)
print(f"###gpus:{gpus}")

CASIA_CLASS_LABELS = ("angry", "fear", "happy", "neutral", "sad", "surprise")#CASIA
EMODB_CLASS_LABELS = ("angry", "boredom", "disgust", "fear", "happy", "neutral", "sad")#EMODB
SAVEE_CLASS_LABELS = ("angry","disgust", "fear", "happy", "neutral", "sad", "surprise")#SAVEE
RAVDE_CLASS_LABELS = ("angry", "calm", "disgust", "fear", "happy", "neutral","sad","surprise")#rav
IEMOCAP_CLASS_LABELS = ("angry", "happy", "neutral", "sad")#iemocap
EMOVO_CLASS_LABELS = ("angry", "disgust", "fear", "happy","neutral","sad","surprise")#emovo
INTERSECT_CLASS_LABELS = ("angry", "happy", "neutral", "sad")
CLASS_LABELS_dict = {"CASIA": CASIA_CLASS_LABELS,
            "EMODB": EMODB_CLASS_LABELS,
            "EMOVO": EMOVO_CLASS_LABELS,
            "IEMOCAP": IEMOCAP_CLASS_LABELS,
            "RAVDE": RAVDE_CLASS_LABELS,
            "SAVEE": SAVEE_CLASS_LABELS,
            "INTERSECT": INTERSECT_CLASS_LABELS}
CLASS_LABELS = CLASS_LABELS_dict[args.data]


class Predict_Demo():
    def __init__(self) -> None:
        mfcc = get_feature("/home/liuhaozhe/signal_processing_projs/collected_audios/recorded_audios/liuhaozhe/liuhaozhe_text1.m4a")
        x_source=np.array([mfcc])
        model = TIMNET_Model(args=args, input_shape=x_source.shape[1:], class_label=CLASS_LABELS)
        x_feats, y_preds = model.infer(x_source, path=args.test_path)
        print(x_feats, y_preds)
