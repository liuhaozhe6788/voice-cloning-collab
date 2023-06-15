import torch
from synthesizer import audio
from synthesizer.hparams import hparams
from synthesizer.models.tacotron import Tacotron, EmotionTacotron
from synthesizer.utils.symbols import symbols
from synthesizer.utils.text import text_to_sequence
from vocoder.display import simple_table
from pathlib import Path
from typing import Union, List
import numpy as np
import librosa


class Synthesizer:
    sample_rate = hparams.sample_rate
    hparams = hparams

    def __init__(self, model_fpath: Path, verbose=True, model_name="Tacotron"):
        """
        The model isn't instantiated and loaded in memory until needed or until load() is called.

        :param model_fpath: path to the trained model file
        :param verbose: if False, prints less information when using the model
        """
        self.model_fpath = model_fpath
        self.verbose = verbose

        # Check for GPU
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        if self.verbose:
            print("Synthesizer using device:", self.device)

        # Tacotron model will be instantiated later on first use.
        self._model = None
        self._model_name = model_name

    def is_loaded(self):
        """
        Whether the model is loaded in memory.
        """
        return self._model is not None

    def load(self):
        """
        Instantiates and loads the model given the weights file that was passed in the constructor.
        """
        if self._model_name is "Tacotron":
            self._model = Tacotron(embed_dims=hparams.tts_embed_dims,
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
                                speaker_embedding_size=hparams.speaker_embedding_size).to(self.device)
        elif self._model_name is "EmotionTacotron":
            self._model = EmotionTacotron(embed_dims=hparams.tts_embed_dims,
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
                                speaker_embedding_size=hparams.speaker_embedding_size,
                                emotion_embedding_size=hparams.emotion_embedding_size).to(self.device)           

        self._model.load(self.model_fpath)
        self._model.eval()

        if self.verbose:
            print("Loaded synthesizer \"%s\" trained to step %d" % (self.model_fpath.name, self._model.state_dict()["step"]))

    def synthesize_spectrograms(self, texts: List[str],
                                speaker_embeddings: Union[np.ndarray, List[np.ndarray]],
                                emotion_embeddings: Union[np.ndarray, List[np.ndarray]],
                                require_visualization=False):
        """
        Synthesizes mel spectrograms from texts and speaker embeddings.

        :param texts: a list of N text prompts to be synthesized
        :param speaker_embeddings: a numpy array or list of speaker embeddings of shape (N, 256)
        :param emotion_embeddings: a numpy array or list of emotion embeddings of shape (N, 39)
        :param require_visualization: if True, a matrix representing the alignments between the
        characters
        and each decoder output step will be returned for each spectrogram
        :return: a list of N melspectrograms as numpy arrays of shape (80, Mi), where Mi is the
        sequence length of spectrogram i, and possibly the alignments.
        """
        # Load the model on the first request.
        if not self.is_loaded():
            self.load()

        # Preprocess text inputs
        inputs = [text_to_sequence(text.strip()) for text in texts]
        if not isinstance(speaker_embeddings, list):
            speaker_embeddings = [speaker_embeddings]

        if not isinstance(emotion_embeddings, list):
            emotion_embeddings = [emotion_embeddings]

        # Batch inputs
        batched_inputs = [inputs[i:i+hparams.synthesis_batch_size]
                             for i in range(0, len(inputs), hparams.synthesis_batch_size)]
        batched_speaker_embeds = [speaker_embeddings[i:i+hparams.synthesis_batch_size]
                             for i in range(0, len(speaker_embeddings), hparams.synthesis_batch_size)]
        batched_emotion_embeds = [emotion_embeddings[i:i+hparams.synthesis_batch_size]
                             for i in range(0, len(emotion_embeddings), hparams.synthesis_batch_size)]
        
        specs = []
        for i, batch in enumerate(batched_inputs, 1):
            if self.verbose:
                print(f"\n| Generating {i}/{len(batched_inputs)}")

            # Pad texts so they are all the same length
            text_lens = [len(text) for text in batch]
            max_text_len = max(text_lens)
            chars = [pad1d(text, max_text_len) for text in batch]
            chars = np.stack(chars)

            # Stack speaker embeddings into 2D array for batch processing
            speaker_embeds = np.stack(batched_speaker_embeds[i-1])

            # Stack emotion embeddings into 2D array for batch processing
            emotion_embeds = np.stack(batched_emotion_embeds[i-1])

            # Convert to tensor
            chars = torch.tensor(chars).long().to(self.device)
            speaker_embeddings = torch.tensor(speaker_embeds).float().to(self.device)
            emotion_embeddings = torch.tensor(emotion_embeds).float().to(self.device)

            # Inference
            _, mels, alignments, stop_tokens = self._model.generate(chars, speaker_embeddings, emotion_embeddings)
            mels = mels.detach().cpu().numpy()
            alignments = alignments.detach().cpu().numpy()
            stop_tokens = stop_tokens.detach().cpu().numpy()
            for m in mels:
                # Trim silence from end of each spectrogram
                while np.max(m[:, -1]) < hparams.tts_stop_threshold:
                    if m.shape[-1] == 1:
                        break
                    m = m[:, :-1]
                # Trim silence from start of each spectrogram
                while np.max(m[:, 0]) < hparams.tts_start_threshold:
                    if m.shape[-1] == 1:
                        break
                    m = m[:, 1:]
                specs.append(m)

        if self.verbose:
            print("\n\nDone.\n")
        return (specs, alignments, stop_tokens) if require_visualization else specs

    @staticmethod
    def load_preprocess_wav(fpath):
        """
        Loads and preprocesses an audio file under the same conditions the audio files were used to
        train the synthesizer.
        """
        wav = librosa.load(str(fpath), hparams.sample_rate)[0]
        if hparams.rescale:
            wav = wav / np.abs(wav).max() * hparams.rescaling_max
        return wav

    @staticmethod
    def make_spectrogram(fpath_or_wav: Union[str, Path, np.ndarray]):
        """
        Creates a mel spectrogram from an audio file in the same manner as the mel spectrograms that
        were fed to the synthesizer when training.
        """
        if isinstance(fpath_or_wav, str) or isinstance(fpath_or_wav, Path):
            wav = Synthesizer.load_preprocess_wav(fpath_or_wav)
        else:
            wav = fpath_or_wav

        mel_spectrogram = audio.melspectrogram(wav, hparams).astype(np.float32)
        return mel_spectrogram

    @staticmethod
    def griffin_lim(mel):
        """
        Inverts a mel spectrogram using Griffin-Lim. The mel spectrogram is expected to have been built
        with the same parameters present in hparams.py.
        """
        return audio.inv_mel_spectrogram(mel, hparams)


def pad1d(x, max_len, pad_value=0):
    return np.pad(x, (0, max_len - len(x)), mode="constant", constant_values=pad_value)
