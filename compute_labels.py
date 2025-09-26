import os
from sqp_snn.data.metrics import generate_metrics_df

data_dir = '/path/to/dataset/'  # Replace with actual dataset path

clean_dir = os.path.join(data_dir, 'clean/')
noisy_dir = os.path.join(data_dir, 'noisy/')

df = generate_metrics_df(clean_dir, noisy_dir, win_len=5)
df.to_pickle(os.path.join(data_dir, 'metrics.pkl'))
