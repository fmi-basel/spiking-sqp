import torch
import numpy as np
import wandb

from types import SimpleNamespace

from sqp_snn.models.spiking_cnn import SpikingCNN

import stork

from sqp_snn.stork_ext.loss import MeanOverTimeMSE
from sqp_snn.optimizers import SMORMS4


def get_sim_timestep(args):
    nb_input_steps  = int(args.input_duration / args.input_timestep)
    sim_timestep = args.input_timestep
    
    return sim_timestep


def get_sim_duration(args):
    sim_timestep = get_sim_timestep(args)
    sim_duration = args.nb_simsteps * sim_timestep
    return sim_duration


def load_model(model_path, args, timestep, device, dtype):
    model = SpikingCNN(args, timestep, device, dtype)
    model.configure(
        input = model.input_group,
        output = model.readout_group,
        loss_stack = MeanOverTimeMSE(),
        generator = stork.generators.StandardGenerator(nb_workers=2, persistent_workers=False),
        optimizer = SMORMS4,
        optimizer_kwargs = dict(lr=args.lr),
        time_step = timestep
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
                          
    return model


def brief_model_summary(model):
    print("\n# Model groups")
    for group in model.groups:
        if group.name is None or group.name == "":
            print(f"no name, {group.shape}")
        else:
            print(f"{group.name}, {group.shape}")


def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal trainable parameters: {total_params:,}")
    if model.wandb is not None:
        model.wandb.run.summary['total_params'] = total_params


def measure_activity(args, model, dataset):
    sim_duration = get_sim_duration(args)
    monitor_indices = []
    num_mon_neurons = 0

    for group in model.groups[2:-1]:
        monitor_indices.append(len(model.monitors))
        model.add_monitor(stork.monitors.SpikeCountMonitor(group))
        print(f"\nMonitor added to {group} (\"{group.name}\")")
        
        n = group.shape[0]
        for dim in group.shape[1:]:
            n *= dim
        num_mon_neurons += n

    print("num_mon_neurons:", num_mon_neurons)
    
    monitors = model.monitor(dataset)

    # Spike metrics per layer averaged over samples
    avg_total_spikes = np.array([])
    avg_spikes_per_neuron = np.array([])

    # Loop over layer-wise monitors
    for idx in monitor_indices:
        monitor = monitors[idx]
        
        # Sum over layer neurons
        spikes_per_sample = monitor.sum(dim=(1,2))
        
        # Average over samples
        avg_total_spikes = np.append(avg_total_spikes, spikes_per_sample.mean().item())
        avg_spikes_per_neuron = np.append(avg_spikes_per_neuron, monitor.mean().item())

    # Sum layers and compute rates
    avg_act_rate = np.sum(avg_total_spikes)/sim_duration
    avg_firing_rate = np.mean(avg_spikes_per_neuron/sim_duration)

    print(f"Avg. neuron rate: {avg_firing_rate:,.2f} Hz")
    
    return avg_act_rate, avg_firing_rate


def load_wandb_model(project_name, run_name, device="cuda:0"):
    api = wandb.Api()
    runs = list(api.runs(project_name, filters={"display_name": run_name}))
    
    if not runs:
        raise ValueError(f"No run found with name {run_name}")
    run = runs[0]
    
    artifacts = run.logged_artifacts()
    model_artifact = next((art for art in artifacts if art.type == 'model'), None)
    if model_artifact is None:
        raise ValueError(f"No model artifact found for run {run_name}")
    
    model_dir = model_artifact.download()
    model_path = f"{model_dir}/final_model.pt"
    
    config = SimpleNamespace(**run.config)
    sim_timestep = get_sim_timestep(config)
    
    model = SpikingCNN(config, sim_timestep, torch.device(device), torch.float)
    brief_model_summary(model)
    model.configure(
        input = model.input_group,
        output = model.readout_group,
        loss_stack = MeanOverTimeMSE(),
        generator = stork.generators.StandardGenerator(nb_workers=2, persistent_workers=False),
        optimizer = SMORMS4,
        optimizer_kwargs = dict(lr=config.lr),
        time_step = sim_timestep
    )
    
    state_dict = torch.load(model_path, map_location=device)    
    model.load_state_dict(state_dict)
    
    
    return model, config


def count_state_flops(model):
    # FLOPs per neuron per time step
    step_flops_lif = 5
    step_flops_adlif = 11

    state_flops = 0

    # Loop over stateful layers (skip input and fanout)
    for group in model.groups[2:]:
        if group.name == 'Encoder':
            step_flops = step_flops_adlif
        else:
            step_flops = step_flops_lif
        
        group_size = 1
        for dim_size in group.shape:
            group_size *= dim_size
        
        state_flops += group_size * step_flops * model.nb_time_steps

    return state_flops
