import torch
import torch.nn.functional as F
from torch import nn


class GlobalAvgPool(nn.Module):
    def forward(self, x):
        return torch.mean(x, dim=(-1, -2))


class GlobalMaxPool(nn.Module):
    def forward(self, x):
        return torch.amax(x, dim=(-1, -2))


class AttentionPooling(nn.Module):
    def __init__(self, hidden_size):
        super(AttentionPooling, self).__init__()
        self.squeeze = nn.Linear(hidden_size, 1)
        self.attention = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # Assuming input x shape is (batch_size, sequence_length, n_features)
        x_squeezed = self.squeeze(x)
        attention_weights = F.softmax(self.attention(x), dim=1)
        x = (x_squeezed * attention_weights).sum(dim=1)
        return x, x_squeezed, attention_weights
    

class AttentionPooling_Framents(AttentionPooling):
    def __init__(self, hidden_size, frame_stride, pool_type):
        super(AttentionPooling_Framents, self).__init__(hidden_size)
        self.frame_stride = frame_stride
        if pool_type == 'max':
            self.frag_pool = nn.MaxPool1d(frame_stride, frame_stride)
        elif pool_type == 'avg':        
            self.frag_pool = nn.AvgPool1d(frame_stride, frame_stride)
        else:
            raise ValueError('Value of pool_type not valid.')

    def forward(self, x):
        # Assuming input x shape is (batch_size, sequence_length, n_features)
        x_squeezed = self.squeeze(x)
        attention_weights = F.softmax(self.attention(x), dim=1).squeeze()
        x_frags = self.frag_pool(x_squeezed.squeeze(-1))
        x_frags_rep = x_frags.repeat_interleave(self.frame_stride, dim=-1)
        x = (x_frags_rep * attention_weights[...,:x_frags_rep.shape[-1]]).sum(dim=1)
        return x, x_frags.unsqueeze(-1), attention_weights.unsqueeze(-1)
    

class AutoPool(nn.Module):
    def __init__(self, pool_dim = 1):
        super(AutoPool, self).__init__()
        self.pool_dim = pool_dim
        self.softmax = nn.Softmax(dim=pool_dim)
        self.register_parameter("alpha", nn.Parameter(torch.ones(1)))

    def forward(self, x):
        weight = self.softmax(torch.mul(x, self.alpha))
        out = torch.sum(torch.mul(x, weight), dim=self.pool_dim)
        return out


class ScoreSigmoid(nn.Module):
    def __init__(self, a_min, a_max, beta):
        super(ScoreSigmoid, self).__init__()
        self.a_min = a_min
        self.a_max = a_max
        self.beta = beta

    def forward(self, x):
        return self.a_min + (self.a_max - self.a_min) * torch.sigmoid(x)
