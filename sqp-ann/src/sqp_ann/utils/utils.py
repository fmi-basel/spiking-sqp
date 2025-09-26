import random
import yaml
import torch
import torchaudio
import numpy as np
import pandas as pd
import os

from neurobench.processors.abstract import NeuroBenchPreProcessor
from omegaconf import OmegaConf
from torchinfo import summary
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
from torchmetrics.functional.regression.mse import mean_squared_error
from torchmetrics.functional.regression.pearson import pearson_corrcoef
from torchmetrics.functional.regression import spearman_corrcoef


def seed_everything(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    os.environ['PYTHONHASHSEED'] = str(seed)


def register_resolvers():
    if not OmegaConf.has_resolver("time_to_samples"):
        OmegaConf.register_new_resolver("time_to_samples", lambda t, sr: int(t * sr))
    if not OmegaConf.has_resolver("nfft_to_bins"):
        OmegaConf.register_new_resolver("nfft_to_bins", lambda n_fft: n_fft // 2 + 1)


def pretty_configs(cfg):
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    cfg_yaml = yaml.dump(cfg_dict, allow_unicode=True, default_flow_style=False)
    return cfg_yaml


def model_summary(model, dataloader):
    dummy_data = next(iter(dataloader))[0]
    original_shape = dummy_data.shape
    
    if hasattr(model, 'preproc'):
        dummy_data = model.preproc(dummy_data)
        
    if len(dummy_data.shape) == 2:
        dummy_data = dummy_data.unsqueeze(1)
    
    try:
        model_summary = summary(
            model,
            input_data=dummy_data,
            verbose=0,
            col_width=20,
            col_names=["input_size",
                       "output_size",
                       "mult_adds",
                       "num_params",
                       "params_percent",
                       "kernel_size",
                       "trainable"],
            #col_names=["input_size", "output_size", "num_params", "mult_adds"],            
            device=next(model.parameters()).device
        )
        return model_summary
    
    except RuntimeError as e:
        print(f"Original input shape: {original_shape}")
        print(f"Preprocessed shape: {dummy_data.shape}")
        print(f"Model: {model.__class__.__name__}")
        raise RuntimeError(f"Model summary failed: {str(e)}")


def compute_metrics(pred_tensor, true_tensor, metrics=["mse", "pcc", "srcc"], colour=None):
    result_metrics = {}
    if "mse" in metrics: 
        result_metrics["mse"] = mean_squared_error(pred_tensor, true_tensor).item()
    if "pcc" in metrics: 
        result_metrics["pcc"] = pearson_corrcoef(pred_tensor, true_tensor).item()
    if "srcc" in metrics: 
        result_metrics["srcc"] = spearman_corrcoef(pred_tensor, true_tensor).item()
    return result_metrics


def train_valid_split(dataset, readers_file, seed, valid_split):
    df_data = dataset.df.reset_index()
    df_readers = pd.read_pickle(readers_file)
    
    reader_groups = df_readers.groupby('reader_id')['file_id'].apply(list).reset_index()
    reader_ids_train, reader_ids_valid = train_test_split(reader_groups['reader_id'], test_size=valid_split, random_state=seed)
    
    file_ids_train = reader_groups[reader_groups['reader_id'].isin(reader_ids_train)]['file_id'].explode().unique()
    file_ids_valid = reader_groups[reader_groups['reader_id'].isin(reader_ids_valid)]['file_id'].explode().unique()
    
    df_train = df_data[df_data['fileid'].isin(file_ids_train)]
    df_valid = df_data[df_data['fileid'].isin(file_ids_valid)]
    
    dataset_train = Subset(dataset, df_train.index.tolist())
    dataset_valid = Subset(dataset, df_valid.index.tolist())
    
    print(f"Validation split complete ({len(dataset_train)}, {len(dataset_valid)}).")
    return dataset_train, dataset_valid


class QnPreProcessor(NeuroBenchPreProcessor):
    def __init__(self, sample_rate=16000, n_fft=640, n_mels=120, device=None):
        super().__init__()
        self.device = device or torch.device('cpu')
        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            n_mels=n_mels,
            power=2,
            center=False
        ).to(self.device)
        self.eps = 1e-9

    def __call__(self, dataset):
        inputs, labels = dataset
        inputs = inputs.to(self.device)
        # Apply mel spectrogram
        melspec = self.melspec(inputs)
        # Apply log and reshape
        log_melspec = torch.log(melspec + self.eps)
        log_melspec = log_melspec.squeeze(1).permute(0, 2, 1)
        return (log_melspec, labels)


class DnsmosPreProcessor(NeuroBenchPreProcessor):
    def __init__(self, sample_rate=16000, n_fft=640, n_mels=120, device=None):
        super().__init__()
        self.device = device or torch.device('cpu')
        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            n_mels=n_mels,
            power=2,
            center=False
        ).to(self.device)
        self.eps = 1e-9

    def __call__(self, dataset):
        inputs, labels = dataset
        inputs = inputs.to(self.device)
        # Apply mel spectrogram
        melspec = self.melspec(inputs)
        # Apply log and reshape
        log_melspec = torch.log(melspec + self.eps)
        return (log_melspec, labels)
