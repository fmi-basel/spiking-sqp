from argparse import ArgumentParser, BooleanOptionalAction

PRESETS = {
    'recurrent': {
        'recurrence': True,
        'nb_conv': 2,
        'block_depth': 1,
        'dropout_conv': 0.4
    },
    'feedforward': {
        'recurrence': False,
        'nb_conv': 4,
        'block_depth': 3,
        'dropout_conv': 0.15
    }
}

def get_args():
    parser = ArgumentParser(description = "Train an SNN for speech quality prediction.")

    parser.add_argument('--model', choices=PRESETS.keys(), default="recurrent")

    # allow overrides (default None so presets can set them)
    parser.add_argument('--recurrence', action=BooleanOptionalAction, default=None)
    parser.add_argument('--nb_conv', type=int, default=None)
    parser.add_argument('--block_depth', type=int, default=None)
    parser.add_argument('--dropout_conv', type=float, default=None)

    # Experiment parameters
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpu', type=int, default=0)
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--val_split', type=float, default=0.05)
    parser.add_argument('--data_hours', type=int, default=50)
    parser.add_argument('--dropout_dense', type=float, default=0)
    
    parser.add_argument('--loss', type=str, default='mse')
    parser.add_argument('--optimizer', type=str, default='smorms4')
    parser.add_argument('--lr', type=float, default=3e-3)
    parser.add_argument('--scheduler', type=str, default=None)
    parser.add_argument('--beta', type=int, default=20)
    
    # Early stopping
    parser.add_argument('--early_stop_metric', type=str, default='pcc')
    parser.add_argument('--early_stop_mode', type=str, default='max')
    parser.add_argument('--patience', type=int, default=25)
    parser.add_argument('--es_min_delta', type=float, default=1e-5)
    
    # Input parameters
    parser.add_argument('--standardize_inputs', action=BooleanOptionalAction, default=False)
    parser.add_argument('--input_duration', type=int, default=5)
    parser.add_argument('--input_timestep', type=float, default=20e-3)

    # Simulation parameters
    parser.add_argument('--nb_simsteps', type=int, default=249)
    parser.add_argument('--nb_freqbands', type=int, default=120)
    parser.add_argument('--stateful', action=BooleanOptionalAction, default=False)

    # Time-constants
    parser.add_argument('--learn_tau_mem', action=BooleanOptionalAction, default=True)
    parser.add_argument('--hetero_tau_mem', action=BooleanOptionalAction, default=True)

    parser.add_argument('--learn_tau_syn', action=BooleanOptionalAction, default=True)
    parser.add_argument('--hetero_tau_syn', action=BooleanOptionalAction, default=True)

    parser.add_argument('--learn_tau_mem_out', action=BooleanOptionalAction, default=True)
    parser.add_argument('--hetero_tau_mem_out', action=BooleanOptionalAction, default=False)

    parser.add_argument('--tau_mem_enc', type=float, default=100e-3)
    parser.add_argument('--tau_syn_enc', type=float, default=20e-3)
    
    parser.add_argument('--tau_mem_hidden', type=float, default=200e-3)
    parser.add_argument('--tau_syn_hidden', type=float, default=40e-3)
    
    parser.add_argument('--tau_mem_readout', type=float, default=500e-3)
    parser.add_argument('--tau_syn_readout', type=float, default=40e-3)

    # Encoder parameters
    parser.add_argument('--encoder_fanout', type=int, default=16)
    parser.add_argument('--encoder_gain', type=int, default=1)

    # Architecture parameters
    parser.add_argument('--neuron_type', type=str, default='lif')
    parser.add_argument('--nb_dense', type=int, default=0)
    parser.add_argument('--global_maxpool', action=BooleanOptionalAction, default=True)

    # Channel parameters
    parser.add_argument('--nb_hidden_channels', type=int, default=32)
    parser.add_argument('--channel_fanout', type=int, default=2)
    parser.add_argument('--increasing_depth', action=BooleanOptionalAction, default=True)
    parser.add_argument('--max_depth', type=int, default=64)

    # Filter parameters
    parser.add_argument('--kernel_size', type=int, default=5)
    parser.add_argument('--stride', type=int, default=2)
    parser.add_argument('--padding', type=int, default=2)
    
    # Regularizers
    parser.add_argument('--pop_ubl2_str', type=float, default=0)
    parser.add_argument('--pop_ubl2_thr', type=float, default=0)
    
    parser.add_argument('--pop_ubl1_str', type=float, default=0)
    parser.add_argument('--pop_ubl1_thr', type=float, default=0)
    
    parser.add_argument('--neur_lbl2_str', type=float, default=100)
    parser.add_argument('--neur_lbl2_thr', type=float, default=1e-3)
    
    parser.add_argument('--neur_ubl2_str', type=float, default=0)
    parser.add_argument('--neur_ubl2_thr', type=float, default=0)

    # Initializer
    parser.add_argument('--sigma_u', type=float, default=1.0)
    parser.add_argument('--nu', type=float, default=5.6)

    # Logging
    parser.add_argument('--logger', type=str, default='none', choices=['wandb', 'none'])
    parser.add_argument('--wandb_project', type=str, default="your-project")
    parser.add_argument('--wandb_entity', type=str, default="your-user")
    parser.add_argument('--log_snapshots', action=BooleanOptionalAction, default=True)
    parser.add_argument('--name', type=str, default=None, help="Optional name for wandb run")

    peek, _ = parser.parse_known_args()
    if peek.model:
        for k, v in PRESETS[peek.model].items():
            if getattr(peek, k) is None:
                parser.set_defaults(**{k: v})
    
    args = parser.parse_args()

    return args
