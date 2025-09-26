# Efficient Streaming Speech Quality Prediction with Spiking Neural Networks

This repository contains the code used for training and evaluating the **convolutional spiking neural networks (CSNNs)** and baseline ANNs presented in our [Interspeech 2025 paper](https://www.isca-archive.org/interspeech_2025/nilsson25_interspeech.html#).

This is a monorepo containing two installable Python packages:

- `sqp-snn`: Training and evaluation of SNNs using [stork](https://github.com/fmi-basel/stork)
- `sqp-ann`: Training and evaluation of baseline ANNs, adapted from our [Interspeech 2024 SQP code](https://github.com/fmi-basel/binary-activation-maps-sqp)

For adapting NeuroBench to stork, we adapted code from [fmi-basel/neural-decoding-RSNN](https://github.com/fmi-basel/neural-decoding-RSNN/tree/main/challenge/neurobench).


## Installation

We used Python 3.10.12 for the development of the code.
We recommend installing each of the two packages **independently** with the following steps:

1. Create and activate a new virtual environment (recommended)
	```bash
	python -m venv sqp-snn-env
	source sqp-snn-env/bin/activate
	```

3. In `spiking-sqp`, navigate to package directory
	```bash
	cd sqp-snn
	```

4. Install requirements
	```bash
	pip install -r requirements.txt
	```

5. Install package:
	```bash
	pip install -e .
	```

6. Repeat the steps above independently for `sqp-ann`


## Dataset

We used the [Interspeech 2020 DNS Challenge dataset](https://github.com/microsoft/DNS-Challenge/tree/interspeech2020/master)
augmented by denoising with [Dynamic NSNet2](https://arxiv.org/abs/2308.16678).

*A link to our augmented dataset will be added here soon.*


### Generate Training Data

You can generate non-augmented training data by following these steps:

1. **Audio data:**
Run the download and single-process data generation scripts from the [Interspeech 2020 DNS Challenge](https://github.com/microsoft/DNS-Challenge/tree/interspeech2020/master).
Change the following parameters in `noisyspeech_synthesizer.cfg` to match our training data prior to augmentation:
	- `audio_length: 5`
	- `total_hours: 50`
	- `snr_lower: -15`
	- `snr_upper: 25`

2. **Labels:**
Compute PESQ speech quality labels by inserting your dataset path in `compute_labels.py` and then running the script
	```bash
	python compute_labels.py
	```

3. Repeat #2 for the test set


### Prepare for training

1. **.env file:**
Create a `.env` file in `sqp-ann/` by copying the provided example
	```bash
	cp sqp-ann/example.env sqp-ann/.env
	```

1. **Paths:**
Update `sqp-snn/src/sqp_snn/project_config.py` and `sqp-ann/.env` with the correct paths for audio data and metrics files (labels)

1. **Reader IDs:**
Generate the `readers.pkl` file used for reader-based validation split by running:
	```bash
	python generate_readers_file.py
	```


## Run experiments

 **CLI note:** `sqp-snn` uses argparse, while `sqp-ann` uses Hydra.
 See examples below for the respective command-line argument formats.

### Train and evaluate CSNNs

**Training time:**
Training a CSNN took up to 9 hours on an NVIDIA A4000 depending on the number of epochs to early stopping on convergence.

In `sqp-snn/`:
- **r-CSNN:**
Train recurrent CSNN
	```bash
	python train.py --model recurrent 
	```

- **ff-CSNN:**
Train feedforward CSNN
	```bash
	python train.py --model feedforward
	```

- **Sparsity experiments:**
Train sequence of models with different degrees of spike-rate regularization yielding the results in Fig. 4 of the paper
	```bash
	bash sparsity_experiments_recurrent.sh
	```
	```bash
	bash sparsity_experiments_feedforward.sh
	```

- **Benchmarking:**
Measure [EFLOPs](http://doi.org/10.1088/2634-4386/addee8) and activation sparsity for a trained CSNN logged to W&B with [NeuroBench](https://neurobench.ai) by running the notebook `benchmark_snn.ipynb`

- **CLI:**
See `src/sqp_snn/args.py` for all possible command-line arguments

- **Logging:**
You can activate W&B logging by setting `--logger` to `wandb` and entering your W&B information in `src/sqp_snn/args.py`


### Train and evaluate baseline ANNs

In `sqp-ann/`:

- **DNSMOS:**
Train a 2D CNN with the architecture of [DNSMOS](https://arxiv.org/abs/2010.15258) (*this trains a CNN from scratch on PESQ-labeled data – it does not imply computing the metric called DNSMOS*)
	```bash
	python train.py model=dnsmos
	```
	- **Benchmarking:**
	Benchmark a trained DNSMOS model with NeuroBench by running the notebook `benchmark_dnsmos.ipynb`

- **Quality-Net:**
Train a causal version of [Quality-Net](http://doi.org/10.21437/Interspeech.2018-1802) based on UniLSTM
	```bash
	python train.py model=qualitynet_realtime
	```
	- **Benchmarking:**
	LSTMs lack activation sparsity and were not supported by NeuroBench at the time of our experiments —
	instead, multiply–accumulate operations are counted with torchinfo by default when you launch training

- **CLI:**
See `sqp-ann/config/train.yaml` for possible command-line arguments

- **Logging:**
Enable W&B logging by uncommenting `- logger_wandb` in `config/train.yaml`
