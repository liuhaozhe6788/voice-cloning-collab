import sys
import traceback
from pathlib import Path
from time import perf_counter as timer
import re

import numpy as np
import torch
import soundfile as sf
import librosa
import spacy

import encoder
from encoder import inference as encoder_infer
from synthesizer.inference import Synthesizer_infer
from synthesizer.utils.cleaners import add_breaks, english_cleaners_predict
from vocoder.display import save_attention_multiple, save_spectrogram, save_stop_tokens
from synthesizer.hparams import syn_hparams
from toolbox.ui import UI
from toolbox.utterance import Utterance
from vocoder import inference as vocoder
from speed_changer.fixSpeed import *
import time


# Use this directory structure for your datasets, or modify it to fit your needs
recognized_datasets = [
    "LibriSpeech/dev-clean",
    "LibriSpeech/dev-other",
    "LibriSpeech/test-clean",
    "LibriSpeech/test-other",
    "LibriSpeech/train-clean-100",
    "LibriSpeech/train-clean-360",
    "LibriSpeech/train-other-500",
    "LibriTTS/dev-clean",
    "LibriTTS/dev-other",
    "LibriTTS/test-clean",
    "LibriTTS/test-other",
    "LibriTTS/train-clean-100",
    "LibriTTS/train-clean-360",
    "LibriTTS/train-other-500",
    "LJSpeech-1.1",
    "VoxCeleb1/wav",
    "VoxCeleb1/test_wav",
    "VoxCeleb2/dev/aac",
    "VoxCeleb2/test/aac",
    "VCTK-Corpus/wav48",
]

# Maximum of generated wavs to keep on memory
MAX_WAVS = 15


class Toolbox:
    def __init__(self, run_id: str, datasets_root: Path, models_dir: Path, seed: int=None):
        sys.excepthook = self.excepthook
        self.datasets_root = datasets_root
        self.utterances = set()
        self.current_generated = (None, None, None, None) # speaker_name, spec, breaks, wav

        self.synthesizer = None # type: Synthesizer_infer
        self.current_wav = None
        self.waves_list = []
        self.waves_count = 0
        self.waves_namelist = []
        self.start_generate_time = None
        self.nlp = spacy.load('en_core_web_sm')

        if not os.path.exists("toolbox_results"):
            os.mkdir("toolbox_results")

        # Check for webrtcvad (enables removal of silences in vocoder output)
        try:
            import webrtcvad
            self.trim_silences = True
        except:
            self.trim_silences = False

        # Initialize the events and the interface
        self.ui = UI()
        self.reset_ui(run_id, models_dir, seed)
        self.setup_events()
        self.ui.start()

    def excepthook(self, exc_type, exc_value, exc_tb):
        traceback.print_exception(exc_type, exc_value, exc_tb)
        self.ui.log("Exception: %s" % exc_value)

    def setup_events(self):
        # Dataset, speaker and utterance selection
        self.ui.browser_load_button.clicked.connect(lambda: self.load_from_browser())
        random_func = lambda level: lambda: self.ui.populate_browser(self.datasets_root,
                                                                     recognized_datasets,
                                                                     level)
        self.ui.random_dataset_button.clicked.connect(random_func(0))
        self.ui.random_speaker_button.clicked.connect(random_func(1))
        self.ui.random_utterance_button.clicked.connect(random_func(2))
        self.ui.dataset_box.currentIndexChanged.connect(random_func(1))
        self.ui.speaker_box.currentIndexChanged.connect(random_func(2))

        # Model selection
        self.ui.encoder_box.currentIndexChanged.connect(self.init_encoder)
        def func():
            self.synthesizer = None
        self.ui.synthesizer_box.currentIndexChanged.connect(func)
        self.ui.vocoder_box.currentIndexChanged.connect(self.init_vocoder)

        # Utterance selection
        func = lambda: self.load_from_browser(self.ui.browse_file())
        self.ui.browser_browse_button.clicked.connect(func)
        func = lambda: self.ui.draw_utterance(self.ui.selected_utterance, "current")
        self.ui.utterance_history.currentIndexChanged.connect(func)
        func = lambda: self.ui.play(self.ui.selected_utterance.wav, Synthesizer_infer.sample_rate)
        self.ui.play_button.clicked.connect(func)
        self.ui.stop_button.clicked.connect(self.ui.stop)
        self.ui.record_button.clicked.connect(self.record)

        #Audio
        self.ui.setup_audio_devices(Synthesizer_infer.sample_rate)

        #Wav playback & save
        func = lambda: self.replay_last_wav()
        self.ui.replay_wav_button.clicked.connect(func)
        func = lambda: self.export_current_wave()
        self.ui.export_wav_button.clicked.connect(func)
        self.ui.waves_cb.currentIndexChanged.connect(self.set_current_wav)

        # Generation
        func = lambda: self.synthesize() or self.vocode()
        self.ui.generate_button.clicked.connect(func)
        self.ui.synthesize_button.clicked.connect(self.synthesize)
        self.ui.vocode_button.clicked.connect(self.vocode)
        self.ui.random_seed_checkbox.clicked.connect(self.update_seed_textbox)

        # UMAP legend
        self.ui.clear_button.clicked.connect(self.clear_utterances)

    def set_current_wav(self, index):
        self.current_wav = self.waves_list[index]

    def export_current_wave(self):
        self.ui.save_audio_file(self.current_wav, Synthesizer_infer.sample_rate)

    def replay_last_wav(self):
        self.ui.play(self.current_wav, Synthesizer_infer.sample_rate)

    def reset_ui(self, run_id: str, models_dir: Path, seed: int=None):
        self.ui.populate_browser(self.datasets_root, recognized_datasets, 0, True)
        self.ui.populate_models(run_id, models_dir)
        self.ui.populate_gen_options(seed, self.trim_silences)

    def load_from_browser(self, fpath=None):
        if fpath is None:
            fpath = Path(self.datasets_root,
                         self.ui.current_dataset_name,
                         self.ui.current_speaker_name,
                         self.ui.current_utterance_name)
            name = str(fpath.relative_to(self.datasets_root))
            speaker_name = self.ui.current_dataset_name + '_' + self.ui.current_speaker_name

            # Select the next utterance
            if self.ui.auto_next_checkbox.isChecked():
                self.ui.browser_select_next()
        elif fpath == "":
            return
        else:
            name = fpath.name
            speaker_name = fpath.parent.name

        # Get the wav from the disk. We take the wav with the vocoder/synthesizer format for
        # playback, so as to have a fair comparison with the generated audio
        wav = Synthesizer_infer.load_preprocess_wav(fpath)
        self.ui.log("Loaded %s" % name)

        self.add_real_utterance(wav, name, speaker_name)

    def record(self):
        wav = self.ui.record_one(encoder_infer.sampling_rate, 5)
        if wav is None:
            return
        self.ui.play(wav, encoder_infer.sampling_rate)

        speaker_name = "user01"
        name = speaker_name + "_rec_%05d" % np.random.randint(100000)
        self.add_real_utterance(wav, name, speaker_name)

    def add_real_utterance(self, wav, name, speaker_name):
        # Compute the mel spectrogram
        spec = Synthesizer_infer.make_spectrogram(wav)
        self.ui.draw_spec(spec, "current")

        path_ori = os.getcwd()
        file_ori = 'temp.wav'
        fpath = os.path.join(path_ori, file_ori)
        sf.write(fpath, wav, samplerate=encoder.params_data.sampling_rate)

        # adjust the speed
        self.wav_ori_info = AudioAnalysis(path_ori, file_ori)
        DelFile(path_ori, '.TextGrid')
        os.remove(fpath)

        # Compute the embedding
        if not encoder_infer.is_loaded():
            self.init_encoder()
        encoder_wav = encoder_infer.preprocess_wav(wav)
        embed, partial_embeds, _ = encoder_infer.embed_utterance(encoder_wav, return_partials=True)
        embed[embed < encoder.params_data.set_zero_thres]=0 # 噪声值置零

        # Add the utterance
        utterance = Utterance(name, speaker_name, wav, spec, embed, partial_embeds, False)
        self.utterances.add(utterance)
        self.ui.register_utterance(utterance)

        # Plot it
        self.ui.draw_embed(embed, name, "current")
        self.ui.draw_umap_projections(self.utterances)
        self.ui.wav_ori_fig.savefig(f"toolbox_results/{name}_info.png", dpi=500)
        if len(self.utterances) >= self.ui.min_umap_points:
            self.ui.umap_fig.savefig(f"toolbox_results/umap_{len(self.utterances)}.png", dpi=500)

    def clear_utterances(self):
        self.utterances.clear()
        self.ui.draw_umap_projections(self.utterances)

    def synthesize(self):
        self.start_generate_time = time.time()
        self.ui.log("Generating the mel spectrogram...")
        self.ui.set_loading(1)

        # Update the synthesizer random seed
        if self.ui.random_seed_checkbox.isChecked():
            seed = int(self.ui.seed_textbox.text())
            self.ui.populate_gen_options(seed, self.trim_silences)
        else:
            seed = None

        if seed is not None:
            torch.manual_seed(seed)

        # Synthesize the spectrogram
        if self.synthesizer is None or seed is not None:
            self.init_synthesizer()

        embed = self.ui.selected_utterance.embed

        def preprocess_text(text):
            text = add_breaks(text) 
            text = english_cleaners_predict(text)
            texts = [i.text.strip() for i in self.nlp(text).sents]  # split paragraph to sentences
            return texts

        texts = preprocess_text(self.ui.text_prompt.toPlainText())
        print(f"the list of inputs texts:\n{texts}")

        embeds = [embed] * len(texts)
        specs, alignments, stop_tokens = self.synthesizer.synthesize_spectrograms(texts, embeds, require_visualization=True)

        breaks = [spec.shape[1] for spec in specs]
        spec = np.concatenate(specs, axis=1)

        save_attention_multiple(alignments, "toolbox_results/attention")
        save_stop_tokens(stop_tokens, "toolbox_results/stop_tokens")

        self.ui.draw_spec(spec, "generated")
        self.current_generated = (self.ui.selected_utterance.speaker_name, spec, breaks, None)
        self.ui.set_loading(0)

    def vocode(self):
        speaker_name, spec, breaks, _ = self.current_generated
        assert spec is not None

        # Initialize the vocoder model and make it determinstic, if user provides a seed
        if self.ui.random_seed_checkbox.isChecked():
            seed = int(self.ui.seed_textbox.text())
            self.ui.populate_gen_options(seed, self.trim_silences)
        else:
            seed = None

        if seed is not None:
            torch.manual_seed(seed)

        # Synthesize the waveform
        if not vocoder.is_loaded() or seed is not None:
            self.init_vocoder()

        def vocoder_progress(i, seq_len, b_size, gen_rate):
            real_time_factor = (gen_rate / Synthesizer_infer.sample_rate) * 1000
            line = "Waveform generation: %d/%d (batch size: %d, rate: %.1fkHz - %.2fx real time)" \
                   % (i * b_size, seq_len * b_size, b_size, gen_rate, real_time_factor)
            self.ui.log(line, "overwrite")
            self.ui.set_loading(i, seq_len)
        if self.ui.current_vocoder_fpath is not None and not self.ui.griffin_lim_checkbox.isChecked():
            self.ui.log("")
            wav = vocoder.infer_waveform(spec, target=vocoder.hp.voc_target, overlap=vocoder.hp.voc_overlap, crossfade=vocoder.hp.is_crossfade, progress_callback=vocoder_progress) 
        else:
            self.ui.log("Waveform generation with Griffin-Lim... ")
            wav = Synthesizer_infer.griffin_lim(spec)
        self.ui.set_loading(0)
        self.ui.log(" Done!", "append")
        self.ui.log(f"Generate time: {time.time() - self.start_generate_time}s")

        # Add breaks
        b_ends = np.cumsum(np.array(breaks) * Synthesizer_infer.hparams.hop_size)
        b_starts = np.concatenate(([0], b_ends[:-1]))
        wavs = [wav[start:end] for start, end, in zip(b_starts, b_ends)]
        breaks = [np.zeros(int(0.15 * Synthesizer_infer.sample_rate))] * len(breaks)
        wav = np.concatenate([i for w, b in zip(wavs, breaks) for i in (w, b)])

        # Trim excessive silences
        if self.ui.trim_silences_checkbox.isChecked():
            wav = encoder_infer.preprocess_wav(wav)

        path_ori = os.getcwd()
        file_ori = 'temp.wav'
        filename = os.path.join(path_ori, file_ori)
        sf.write(filename, wav.astype(np.float32), syn_hparams.sample_rate)
        self.ui.log("\nSaved output (haven't change speed) as %s\n\n" % filename)

        # Fix Speed(generate new audio)
        fix_file, speed_factor = work(*self.wav_ori_info, filename)
        self.ui.log(f"\nSaved output (fixed speed) as {fix_file}\n\n")
        wav, _ = librosa.load(fix_file, syn_hparams.sample_rate)
        os.remove(fix_file)

        # Play it
        wav = wav / np.abs(wav).max() * 4
        self.ui.play(wav, Synthesizer_infer.sample_rate)

        # Name it (history displayed in combobox)
        # TODO better naming for the combobox items?
        wav_name = str(self.waves_count + 1)

        #Update waves combobox
        self.waves_count += 1
        if self.waves_count > MAX_WAVS:
          self.waves_list.pop()
          self.waves_namelist.pop()
        self.waves_list.insert(0, wav)
        self.waves_namelist.insert(0, wav_name)

        self.ui.waves_cb.disconnect()
        self.ui.waves_cb_model.setStringList(self.waves_namelist)
        self.ui.waves_cb.setCurrentIndex(0)
        self.ui.waves_cb.currentIndexChanged.connect(self.set_current_wav)

        # Update current wav
        self.set_current_wav(0)

        #Enable replay and save buttons:
        self.ui.replay_wav_button.setDisabled(False)
        self.ui.export_wav_button.setDisabled(False)

        # Compute the embedding
        # TODO: this is problematic with different sampling rates, gotta fix it
        if not encoder_infer.is_loaded():
            self.init_encoder()
        encoder_wav = encoder_infer.preprocess_wav(wav)
        embed, partial_embeds, _ = encoder_infer.embed_utterance(encoder_wav, return_partials=True)

        # Add the utterance
        name = speaker_name + "_gen_%05d_" % np.random.randint(100000) + str(speed_factor)
        utterance = Utterance(name, speaker_name, wav, spec, embed, partial_embeds, True)
        self.utterances.add(utterance)

        # Plot it
        self.ui.draw_embed(embed, name, "generated")
        self.ui.draw_umap_projections(self.utterances)
        self.ui.wav_gen_fig.savefig(f"toolbox_results/{name}_info.png", dpi=500)
        if len(self.utterances) >= self.ui.min_umap_points:
            self.ui.umap_fig.savefig(f"toolbox_results/umap_{len(self.utterances)}.png", dpi=500)

    def init_encoder(self):
        model_fpath = self.ui.current_encoder_fpath

        self.ui.log("Loading the encoder %s... " % model_fpath)
        self.ui.set_loading(1)
        start = timer()
        encoder_infer.load_model(model_fpath)
        self.ui.log("Done (%dms)." % int(1000 * (timer() - start)), "append")
        self.ui.set_loading(0)

    def init_synthesizer(self):
        model_fpath = self.ui.current_synthesizer_fpath

        self.ui.log("Loading the synthesizer %s... " % model_fpath)
        self.ui.set_loading(1)
        start = timer()
        self.synthesizer = Synthesizer_infer(model_fpath)
        self.ui.log("Done (%dms)." % int(1000 * (timer() - start)), "append")
        self.ui.set_loading(0)

    def init_vocoder(self):
        model_fpath = self.ui.current_vocoder_fpath
        # Case of Griffin-lim
        if model_fpath is None:
            return

        self.ui.log("Loading the vocoder %s... " % model_fpath)
        self.ui.set_loading(1)
        start = timer()
        vocoder.load_model(model_fpath)
        self.ui.log("Done (%dms)." % int(1000 * (timer() - start)), "append")
        self.ui.set_loading(0)

    def update_seed_textbox(self):
       self.ui.update_seed_textbox()
