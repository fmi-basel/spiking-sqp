import numpy as np
import torch
import torch.nn as nn

from torchmetrics.functional.regression.pearson import pearson_corrcoef
from torchmetrics.functional.regression import spearman_corrcoef

from stork.loss_stacks import LossStack


class MeanOverTimeMSE(LossStack):

    def __init__(self, time_dimension=1):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.time_dim = time_dimension

    def get_metric_names(self):
        return ["pcc", "srcc"]

    def compute_loss(self, output, target):
        """ Computes MSE loss between output and target. """
        mot = torch.mean(output, self.time_dim)
        target = target.unsqueeze(1).float()
        
        loss_value = self.mse_loss(mot, target)
        pcc = pearson_corrcoef(mot, target)
        srcc = spearman_corrcoef(mot, target)
        
        self.metrics = [pcc.item(), srcc.item()]
        return loss_value

    def predict(self, output):
        return torch.mean(output, self.time_dim)
    
    def __call__(self, output, targets):
        return self.compute_loss(output, targets)
