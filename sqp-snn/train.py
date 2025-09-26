import matplotlib.pyplot as plt
import os
import pandas as pd
import random
import torch
import wandb

from datetime import datetime

import stork
from stork.datasets import DatasetView
from stork.generators import StandardGenerator
from stork.plotting import plot_activity_snapshot

from sqp_snn import project_config
from sqp_snn.args import get_args

from sqp_snn.data.utils import validation_split, subsample_dataset
from sqp_snn.data.datasets import SQPDataset

from sqp_snn.models.utils import get_sim_timestep, get_sim_duration, count_parameters, measure_activity
from sqp_snn.models.spiking_cnn import SpikingCNN

from sqp_snn.stork_ext.early_stopper import EarlyStopper
from sqp_snn.stork_ext.loss import MeanOverTimeMSE

from sqp_snn.training import seed_everything, get_optimizer, get_lr_scheduler
from sqp_snn.visualization import preds_targets_plot


args = get_args()
seed_everything(args.seed)
timestring = datetime.now().strftime('%Y-%m-%d/%H-%M-%S')
output_path = os.path.join(project_config.BASE_PATH, 'outputs', timestring)
os.makedirs(output_path, exist_ok=True)

sim_timestep = get_sim_timestep(args)
sim_duration = get_sim_duration(args)


# Load dataset

df_mother = pd.read_pickle(project_config.METRICS_FILE_TRAIN)
dataset_mother = SQPDataset(
    df = df_mother,
    #cache_fname = "%s/dns2020_sqp_train_cache.pkl.gz"%(project_config.CACHE_DIR)
)

dataset_train, dataset_valid, df_train, df_valid = validation_split(dataset_mother,
                                                                    project_config.READERS_FILE_TRAIN,
                                                                    args.seed,
                                                                    args.val_split)

df_test = pd.read_pickle(project_config.METRICS_FILE_TEST)
dataset_test = SQPDataset(
    df = df_test,
    #cache_fname = "%s/dns2020_sqp_test_cache.pkl.gz"%(project_config.CACHE_DIR)
)


# Setup

device = torch.device(f"cuda:{args.gpu}")
dtype = torch.float
loss_stack = MeanOverTimeMSE()

early_stopper = EarlyStopper(patience=args.patience,
                             min_delta=args.es_min_delta,
                             mode=args.early_stop_mode,
                             verbose=True)

optimizer, optimizer_kwargs = get_optimizer(args)
scheduler, scheduler_kwargs = get_lr_scheduler(args)
generator = StandardGenerator(nb_workers=2, persistent_workers=False)

model = SpikingCNN(args, sim_timestep, device, dtype)

model.add_monitor(stork.monitors.SpikeCountMonitor(model.groups[2]))
print("Monitor added for " + str(model.groups[2]) + '("' + model.groups[2].name +'")')


if args.logger == "wandb":
    wandb.init(project = args.wandb_project,
               name = args.name,
               entity = args.wandb_entity,
               config = vars(args))
elif args.logger == "none":
    wandb = None

model.configure(input = model.input_group,
                output = model.readout_group,
                loss_stack = loss_stack,
                generator = generator,
                optimizer = optimizer,
                optimizer_kwargs = optimizer_kwargs,
                scheduler = scheduler,
                scheduler_kwargs = scheduler_kwargs,
                time_step = sim_timestep,
                wandb = wandb)

count_parameters(model)
model.summary()

monitor = model.monitor(DatasetView(dataset_train,
                                    random.sample(range(len(dataset_train)), min(len(dataset_train),1000))))
avg_encoder_spikes = monitor[0].mean().item()/sim_duration
args.nu = avg_encoder_spikes
print("Encoder nu = ", avg_encoder_spikes)

plt.figure(dpi=150)
plot_activity_snapshot(model,
                       data = DatasetView(dataset_train, random.sample(range(len(dataset_train)), 5)),
                       nb_samples = 5,
                       point_alpha = 0.3,
                       show_input_class = False,
                       input_heatmap = True)

if args.logger == "wandb":
    wandb.log({"snapshot_init": wandb.Image(plt)})


# Train

results = {}
history = model.fit_validate(dataset_train,
                             dataset_valid,
                             nb_epochs = args.epochs,
                             early_stopper = early_stopper,
                             early_stop_metric = args.early_stop_metric,
                             verbose = True)

results["train_loss"] = history["loss"].tolist()
results["train_pcc"] = history["pcc"].tolist()
results["train_srcc"] = history["srcc"].tolist()

results["valid_loss"] = history["val_loss"].tolist()
results["valid_pcc"] = history["val_pcc"].tolist()
results["valid_srcc"] = history["val_srcc"].tolist()

valid_act_rate, valid_firing_rate = measure_activity(args, model, dataset_valid)
if args.logger == "wandb":
    model.wandb.run.summary['valid_act_rate'] = valid_act_rate
    model.wandb.run.summary['valid_firing_rate'] = valid_firing_rate


# Test

scores = model.evaluate(dataset_test).tolist()
results["test_loss"], _, results["test_pcc"], results["test_srcc"] = scores
if args.logger == "wandb":
    wandb.run.summary["test_loss"] = results["test_loss"]
    wandb.run.summary["test_pcc"]  = results["test_pcc"]
    wandb.run.summary["test_srcc"]  = results["test_srcc"]

test_act_rate, test_firing_rate = measure_activity(args, model, dataset_test)
if args.logger == "wandb":
    model.wandb.run.summary['test_act_rate'] = test_act_rate
    model.wandb.run.summary['test_firing_rate'] = test_firing_rate


# Post-training plots

plt.figure(dpi=150)
plot_activity_snapshot(model,
                       data = DatasetView(dataset_train, random.sample(range(len(dataset_train)), 5)),
                       nb_samples = 5,
                       point_alpha = 0.3,
                       show_input_class = False,
                       input_heatmap = True)
if args.logger == "wandb":
    wandb.log({"snapshot_final": wandb.Image(plt)})

plt.figure()
preds_targets_plot(model,
                   DatasetView(dataset_train, random.sample(range(len(dataset_train)), min(len(dataset_train), 600))),
                   "train")
if args.logger == "wandb":
    wandb.log({"preds_targets_train": wandb.Image(plt)})

plt.figure()
preds_targets_plot(model,
                   DatasetView(dataset_valid, random.sample(range(len(dataset_valid)), min(len(dataset_valid), 600))),
                   "valid")
if args.logger == "wandb":
    wandb.log({"preds_targets_valid": wandb.Image(plt)})

plt.figure()
preds_targets_plot(model, dataset_test, "test")
if args.logger == "wandb":
    wandb.log({"preds_targets_test": wandb.Image(plt)})


# Save model

model_path = os.path.join(output_path, 'final_model.pt')
torch.save(model.state_dict(), model_path)

if args.logger == "wandb":
    model_artifact = wandb.Artifact('final-model', type='model')
    model_artifact.add_file(model_path)
    wandb.log_artifact(model_artifact)
    wandb.finish()
