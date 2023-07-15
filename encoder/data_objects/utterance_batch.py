from pathlib import Path
import numpy as np
from typing import List
from encoder.data_objects.utterance import Utterance


class UtteranceBatch:
    def __init__(self, utterance_path: List[Path], n_frames: int):
        self.utterance = Utterance(utterance_path, None) 
        self.data = np.array(self.utterance.random_partial(n_frames)[0])