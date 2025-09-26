import numpy as np
import os
import pandas as pd
import re
import torch
import torchaudio

from tqdm import tqdm
from stork import utils
import torchaudio.transforms as T
from pathlib import Path

from sqp_snn import project_config


class RawAudioDataset:
    """ Provides a baseclass for audio datasets. 

    Args:
        nb_fft: The number of FFT channels to compute
        nb_filt: The number of mel-spaced filters to compute in the filter bank
    """

    def __init__(self,
                 nb_fft = 640,
                 nb_filt = 120,
                 power = 2):

        self.nb_fft = nb_fft
        self.nb_filt = nb_filt
        self.power = power

        self.data = []
        self.signal_lengths = []
        
    def recursive_walk(self, rootdir):
        """
        Yields:
            str: All filnames in rootdir, recursively.
        """
        for r, dirs, files in os.walk(rootdir):
            for f in files:
                yield os.path.join(r, f)

    def get_signal(self, fname):
        """ Returns audio data and sampling rate from audio file. """
        signal, sample_rate = torchaudio.load(fname)
        print(sample_rate)
        return signal, sample_rate

    def get_feature(self, signal, sample_rate):
        """ Compute mel-spaced feature from audio data file. 

        Args:
            fname (str): The file name of the audio file to operate on

        Returns: 
            An audio feature
        """
        filter_banks = T.MelSpectrogram(sample_rate=sample_rate,
                                        n_fft=self.nb_fft,
                                        n_mels=self.nb_filt,
                                        power=self.power,
                                        center=False
                                        )(signal)
        filter_banks = torch.log(filter_banks + 1e-9)
        filter_banks = filter_banks.squeeze(0)
        filter_banks = filter_banks.T

        return filter_banks

    def __len__(self):
        return len(self.data)

    def prepare_item(self, index):
        raise NotImplemented

    def __getitem__(self, index):
        return self.data[index]

    def cache(self, fname):
        """ Save current data to cache file. """
        utils.write_to_file(self.data, fname)

    def load_cached(self, fname):
        """ Load from cached data file. """
        self.data = utils.load_from_file(fname)
        

class SQPDataset(RawAudioDataset):
    def __init__(self,
                 df = None,
                 sr = 16000,
                 metric = "pesq",
                 cache_fname = None,
                 exclude_degr_types = None):
        super().__init__()
        assert os.path.isdir(project_config.DATASET_DIR), f"Rootdir {project_config.DATASET_DIR} does not exist!"
        # store arguments
        self.df = df
        self.rootdir = Path(project_config.DATASET_DIR)
        self.sr = sr
        self.metric = metric
        self.winlen = 5
        self.winhop = self.winlen
        self.exclude_degr_types = exclude_degr_types or []

        
        if cache_fname is None:
            print("Loading data...")
            self.load_data()
            print(f"Loaded {len(self.data)} data...")
        else:
            try:
                print("Trying to load cached data...")
                self.load_cached(cache_fname)
                print(f"Finished loading {len(self.data)} cached data.")
            except FileNotFoundError:
                print("Cache file not found")
                print("Loading raw data...")
                self.load_data()
                self.cache(cache_fname)

    def load_data(self):
        df = self.df[~self.df['degr_type'].isin(self.exclude_degr_types)]
        df = df.dropna()
        self.data = [self.prepare_item(idx, row) for idx, row in tqdm(df.iterrows(), total=len(df))]

    def prepare_item(self, idx, row):
        original_degr_path = str(row['degr_path']).strip()
        degr_fpath = self._resolve_path(original_degr_path)
        label = row[self.metric]
        fname = os.path.join(self.rootdir, degr_fpath)
        segment_id = row['segment'] if 'segment' in row.index else 0
        signal, sample_rate = self.get_signal(fname, segment_id)
        feat = self.get_feature(signal, sample_rate)
        return feat, label
    
    def get_signal(self, fname, segment_id):
        num_frames = int(self.winlen * self.sr)
        offset = int(segment_id * self.winhop * self.sr)
        signal, sample_rate = torchaudio.load(fname,
                                              frame_offset=offset,
                                              num_frames=num_frames)
        assert sample_rate == self.sr, f"Mismatch between expected sample rate and input data sample rate: {self.sr} != {sample_rate}"
        return signal, sample_rate
    
    def _resolve_path(self, original_path):
        p = Path(original_path)

        if p.is_absolute() and p.exists():
            return p

        candidate = self.rootdir / p
        if candidate.exists():
            return candidate

        # Longest-suffix search under rootdir
        parts = p.parts
        for i in range(len(parts)):
            candidate = self.rootdir.joinpath(*parts[i:])
            if candidate.exists():
                return candidate

        raise FileNotFoundError(
            f"Could not locate audio file for '{original_path}'. "
            f"Tried '{self.rootdir / Path(original_path)}' and suffixes under '{self.rootdir}'."
        )
