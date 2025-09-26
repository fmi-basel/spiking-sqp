import numpy as np
import os
import pandas as pd

from scipy import stats
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from stork.datasets import DatasetView

from sqp_snn.project_config import SAMPLE_LENGTH


def check_nans(df, stage_name):
    nan_counts = df.isnull().sum()
    if nan_counts.sum() > 0:
        print(f"NaN values found in {stage_name}:")
        print(nan_counts[nan_counts > 0])
    else:
        print(f"No NaN values found in {stage_name}")
    print()


def validation_split(dataset, readers_file, seed, val_split=0.05):
    """Split data by speaker IDs.
    """
    df_data = dataset.df.reset_index()
    df_readers = pd.read_pickle(readers_file)
    reader_groups = df_readers.groupby('reader_id')['file_id'].apply(list).reset_index()
    
    reader_ids_train, reader_ids_valid = train_test_split(reader_groups['reader_id'],
                                                          test_size = val_split,
                                                          random_state = seed)
    
    file_ids_train = reader_groups[reader_groups['reader_id'].isin(reader_ids_train)]['file_id'].explode().unique()
    file_ids_valid = reader_groups[reader_groups['reader_id'].isin(reader_ids_valid)]['file_id'].explode().unique()
    
    df_train = df_data[df_data['fileid'].isin(file_ids_train)]
    df_valid = df_data[df_data['fileid'].isin(file_ids_valid)]
    
    dataset_train = DatasetView(dataset, df_train.index.tolist())
    dataset_valid = DatasetView(dataset, df_valid.index.tolist())
    
    print(f"Validation split complete ({len(dataset_train)}, {len(dataset_valid)}).")
    
    return dataset_train, dataset_valid, df_train, df_valid


def subsample_dataset(data_hours, metrics_file, seed, excluded=['dynnsnet2_exit_2']):
    """
    Returns a subsampled dataset, selecting one degradation type per file in each pass.
    Continues passes until target hours are reached or all combinations are exhausted.
    """
    np.random.seed(seed)
    base = metrics_file.rsplit('.pkl', 1)[0]
    metrics_file_sub = f"{base}_{data_hours}h.pkl"

    if os.path.exists(metrics_file_sub):
        print("Loading existing subsampled dataframe...")
        return pd.read_pickle(metrics_file_sub)

    print("Generating subsampled dataframe...")
    df = pd.read_pickle(metrics_file)
    df = df[~df['degr_type'].isin(excluded)].dropna()

    total_rows = len(df)
    available_hours = (total_rows * SAMPLE_LENGTH) / 3600

    rows_target = int((data_hours * 3600) / SAMPLE_LENGTH)

    file_ids = df['fileid'].unique()
    selected_rows = []
    sampled_combinations = set()

    pass_count = 0
    while len(selected_rows) < rows_target:
        pass_count += 1
        print(f"Pass {pass_count}")
        
        np.random.shuffle(file_ids)
        files_sampled_this_pass = set()

        for file_id in tqdm(file_ids):
            if len(selected_rows) >= rows_target:
                break
            
            if file_id in files_sampled_this_pass:
                continue

            file_rows = df[df['fileid'] == file_id]
            available_degr_types = [dt for dt in file_rows['degr_type'].unique() 
                                    if (file_id, dt) not in sampled_combinations]

            if not available_degr_types:
                continue

            degr_type = np.random.choice(available_degr_types)
            selected_row = file_rows[file_rows['degr_type'] == degr_type].iloc[0]
            selected_rows.append(selected_row)
            sampled_combinations.add((file_id, degr_type))
            files_sampled_this_pass.add(file_id)

        if len(files_sampled_this_pass) == 0:
            print("All possible combinations have been sampled. Stopping.")
            break

    df_sub = pd.DataFrame(selected_rows)
    hours_sampled = (len(df_sub) * SAMPLE_LENGTH) / 3600
    print(f"Sampled {hours_sampled:.2f} hours of data from {len(df_sub['fileid'].unique())} unique files.")
    pd.to_pickle(df_sub, metrics_file_sub)
    return df_sub
