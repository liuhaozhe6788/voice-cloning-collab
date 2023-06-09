from vocoder.models.fatchord_version import WaveRNN
from vocoder import hparams as hp
from scipy.fft import rfft, rfftfreq
from scipy import signal
from denoiser.pretrained import master64
import librosa
import numpy as np
import torch
import torchaudio
import noisereduce as nr    


_model = None   # type: WaveRNN

def load_model(weights_fpath, verbose=True):
    global _model, _device
    
    if verbose:
        print("Building Wave-RNN")
    _model = WaveRNN(
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
        _model = _model.cuda()
        _device = torch.device('cuda')
    else:
        _device = torch.device('cpu')
    
    if verbose:
        print("Loading model weights at %s" % weights_fpath)
    checkpoint = torch.load(weights_fpath, _device)
    _model.load_state_dict(checkpoint['model_state'])
    _model.eval()


def is_loaded():
    return _model is not None


def infer_waveform(mel, normalize=True,  batched=True, target=8000, overlap=800, 
                   progress_callback=None):
    """
    Infers the waveform of a mel spectrogram output by the synthesizer (the format must match 
    that of the synthesizer!)
    
    :param normalize:  
    :param batched: 
    :param target: 
    :param overlap: 
    :return: 
    """
    if _model is None:
        raise Exception("Please load Wave-RNN in memory before using it")
    
    if normalize:
        mel = mel / hp.mel_max_abs_value
    mel = torch.from_numpy(mel[None, ...])
    wav = _model.generate(mel, batched, target, overlap, hp.mu_law, progress_callback)
    wav = waveform_denoising(wav)
    return wav

def waveform_denoising(wav):
    prop_decrease = hp.prop_decrease_low_freq if hp.sex else hp.prop_decrease_high_freq
    if torch.cuda.is_available():
        _device = torch.device('cuda')
    else:
        _device = torch.device('cpu')
    model = master64().to(_device)
    noisy=torch.from_numpy(np.array([wav])).to(_device).float()
    estimate = model(noisy)[0].cpu().detach().numpy()
    return  nr.reduce_noise(np.squeeze(estimate), hp.sample_rate, prop_decrease=prop_decrease) 

def get_dominant_freq(wav, name="fft"):
    import matplotlib.pyplot as plt
    N = len(wav)
    fft_wav = rfft(wav)
    fft_freq = np.real(rfftfreq(N, 1 / hp.sample_rate))
    fft_least_index = np.where(fft_freq >= 60)[0][0]
    fft_max = max(fft_wav[fft_least_index: ])
    fft_max_index = np.where(fft_wav == fft_max)[0][0]
    fft_max_freq = fft_freq[fft_max_index]
    # plt.clf()
    # plt.plot(fft_freq, fft_wav)
    # plt.savefig(f"{name}.png", dpi=300)
    return fft_max_freq