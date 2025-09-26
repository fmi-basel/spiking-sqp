import numpy as np
import os
import random
import torch

from stork.optimizers import SMORMS3
from sqp_snn.optimizers import SMORMS4


def seed_everything(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    os.environ['PYTHONHASHSEED'] = str(seed)


def get_optimizer(args):
    
    if args.optimizer == 'smorms3':
        optimizer = SMORMS3
    elif args.optimizer == 'smorms4':
        optimizer = SMORMS4
    else:
        raise ValueError(f"Invalid optimizer '{args.optimizer}'.")

    optimizer_kwargs = dict(lr=args.lr)
    
    return optimizer, optimizer_kwargs


def get_lr_scheduler(args):
    
    if args.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR
        scheduler_kwargs = {
            'T_max': 50,
            'eta_min': args.lr * 1e-3}
    elif args.scheduler == 'cosine_warm_restarts':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts
        scheduler_kwargs = {
            'T_0': 20,
            'T_mult': 2,
            'eta_min': args.lr * 0.001,
            'last_epoch': -1}
    elif args.scheduler == 'reduce_on_plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau
        scheduler_kwargs = {
            'mode': 'min',
            'factor': 0.1,
            'patience': 15,
            'threshold': 0.0001,
            'min_lr': 1e-6}
    else:
        scheduler = None
        scheduler_kwargs = {}
        
    return scheduler, scheduler_kwargs
