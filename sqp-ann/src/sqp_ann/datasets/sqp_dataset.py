import os
import re
import logging
import torch
import torchaudio
import pandas as pd
from torch.nn import functional as F
from pathlib import Path

logger = logging.getLogger(__name__)


class SQPDataset():
    def __init__(self, audio_dir, metrics_file, sr, winlen, winhop, metric):
        self.winlen = winlen
        self.winhop = winhop
        self.sr = sr
        self.metric = metric
        self.audio_dir = Path(audio_dir)
        self.df = pd.read_pickle(metrics_file)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        metric = torch.tensor(row[self.metric], dtype=torch.float32)
        
        original_degr_path = str(row['degr_path']).strip()
        degr_path = self._resolve_path(original_degr_path)
        
        segment_id = row['segment'] if 'segment' in row.index else 0
        noisy_audio = self._load_audio(degr_path, segment_id)
        return noisy_audio, metric

    def __len__(self):
        return len(self.df)

    def _resolve_path(self, original_path):
        p = Path(original_path)

        if p.is_absolute() and p.exists():
            return p

        candidate = self.audio_dir / p
        if candidate.exists():
            return candidate

        # Longest-suffix search under audio_dir
        parts = p.parts
        for i in range(len(parts)):
            candidate = self.audio_dir.joinpath(*parts[i:])
            if candidate.exists():
                return candidate

        raise FileNotFoundError(
            f"Could not locate audio file for '{original_path}'. "
            f"Tried '{self.audio_dir / Path(original_path)}' and suffixes under '{self.audio_dir}'."
        )    

    def _load_audio(self, fpath, segment_id):
        num_frames = int(self.winlen * self.sr)
        offset = int(segment_id * self.winhop * self.sr)
        out, sr = torchaudio.load(fpath, frame_offset=offset, num_frames=num_frames)
        assert sr == self.sr, f"Mismatch between expected sample rate and input data sample rate: {self.sr} != {sr}"
        return out
