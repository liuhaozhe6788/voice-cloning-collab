import librosa
import numpy as np

def get_mfcc(filepath_or_wav, fs=None, mfcc_len: int = 39, mean_signal_length: int = 100000):
    if isinstance(filepath_or_wav, str):
        signal, fs = librosa.load(filepath_or_wav)
    else:
        signal = filepath_or_wav
    s_len = len(signal)

    if s_len < mean_signal_length:
        pad_len = mean_signal_length - s_len
        pad_rem = pad_len % 2
        pad_len //= 2
        signal = np.pad(signal, (pad_len, pad_len + pad_rem), 'constant', constant_values = 0)
    else:
        pad_len = s_len - mean_signal_length
        pad_len //= 2
        signal = signal[pad_len:pad_len + mean_signal_length]
    mfcc = librosa.feature.mfcc(y=signal, sr=fs, n_mfcc=mfcc_len)
    mfcc = mfcc.T
    feature = mfcc
    return feature

if __name__ == "__main__":
    mfcc = get_mfcc("/home/liuhaozhe/signal_processing_projs/collected_audios/recorded_audios/liuhaozhe/liuhaozhe_text1.m4a")
    print(mfcc.shape)