import librosa as lr
import os
import pandas as pd
import re
import torch

from tqdm import tqdm

from torchmetrics.functional.audio.pesq import perceptual_evaluation_speech_quality as pesq


def compute_metrics(clean_file, degraded_file, win_len=5):
    
    sig_clean, sr_clean = lr.load(clean_file, sr=None)
    sig_degr, sr_degr = lr.load(degraded_file, sr=None)
    
    if sr_clean != sr_degr:
        raise ValueError("Sample rates do not match")
    
    win_samples = lr.time_to_samples(win_len, sr=sr_degr)
    
    clean_frame = torch.tensor(sig_clean[:win_samples])
    degr_frame = torch.tensor(sig_degr[:win_samples])
        
    try:
        metrics = {'pesq': pesq(degr_frame, clean_frame, fs=sr_degr, mode="wb").item()}
    except Exception as e:
        metrics = None

    return metrics


def extract_file_id(filename):
    match = re.search(r'fileid_(\d+)', filename)
    if match:
        return int(match.group(1))
    raise ValueError(f"Could not extract file ID from filename: {filename}")


def find_matching_file(directory, file_id):
    target = f'fileid_{file_id}.wav'
    for filename in os.listdir(directory):
        if filename.endswith(target):
            return os.path.join(directory, filename)
    return None


def generate_metrics_df(clean_dir, noisy_dir, processed_dirs=None, win_len=5):
    results = []
    clean_files = [f for f in os.listdir(clean_dir) if f.endswith('.wav')]
    clean_files.sort(key=lambda x: extract_file_id(x))
    
    for filename in tqdm(clean_files):
        clean_file = os.path.join(clean_dir, filename)
        file_id = extract_file_id(filename)
        
        # Get metrics for noisy files
        noisy_file = find_matching_file(noisy_dir, file_id)
        if noisy_file:
            metrics = compute_metrics(clean_file, noisy_file, win_len)
            if metrics is None:
                continue
            results.append({
                'fileid': file_id,
                'degr_type': 'noisy',
                'clean_path': clean_file,
                'degr_path': noisy_file,
                'pesq': metrics['pesq']
            })
        
        # Get metrics for files processed with DynNSNet2
        for i, processed_dir in enumerate(processed_dirs or []):
            processed_file = find_matching_file(processed_dir, file_id)
            if processed_file:
                metrics = compute_metrics(clean_file, processed_file, win_len)
                if metrics is None:
                    continue
                results.append({
                    'fileid': file_id,
                    'degr_type': f'dynnsnet2_exit_{i}',
                    'clean_path': clean_file,
                    'degr_path': processed_file,
                    'pesq': metrics['pesq'],
                })

    return pd.DataFrame(results)
