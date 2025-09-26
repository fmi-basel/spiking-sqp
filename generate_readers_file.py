import matplotlib.pyplot as plt
import os
import pandas as pd
import re

from sqp_snn.project_config import METRICS_FILE_TRAIN, DATASET_DIR

df = pd.read_pickle(METRICS_FILE_TRAIN)

noisy_df = df[df['degr_type'] == 'noisy']

reader_ids = []
file_ids = []

for index, row in noisy_df.iterrows():
    match = re.search(r'reader_(\d+)', row['degr_path'])
    if match is not None:
        reader_id = match.group(1)
        file_id = row['fileid']
        reader_ids.append(reader_id)
        file_ids.append(file_id)

readers_df = pd.DataFrame({'file_id': file_ids,
                           'reader_id': reader_ids})

readers_file = os.path.join(DATASET_DIR, 'dns2020_sqp/readers.pkl')
readers_df.to_pickle(readers_file)
