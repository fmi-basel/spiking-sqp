import torch
from torch import nn
from sqp_ann.models.blocks import GlobalAvgPool, ScoreSigmoid


class QualityNet(nn.Module):
    def __init__(self, n_fft, n_mels, hidden_rnn, hidden_fc, last_activation, preproc, bidirectional=True):
        super(QualityNet, self).__init__()
        self.n_fft = n_fft
        self.n_mels = n_mels
        # layers
        self.rnn = nn.LSTM(
            input_size=n_mels, 
            hidden_size=hidden_rnn,
            bidirectional=bidirectional,
            num_layers=1, 
            batch_first=True)
        
        self._initialize_forget_gate_bias(hidden_rnn, bidirectional, forget_bias=-3.0)
        
        rnn_out_size = hidden_rnn * 2 if bidirectional else hidden_rnn
        
        self.fc = nn.Sequential(nn.Linear(rnn_out_size, hidden_fc),
                                nn.ELU(),
                                nn.Linear(hidden_fc, hidden_fc),
                                nn.ELU(),
                                nn.Linear(hidden_fc, 1))
        self.pool = GlobalAvgPool()
        self.last_activation = self._pick_last_activation(last_activation)
        self.preproc = preproc
        self.eps = 1e-9

    def loss_function(self, mos_pred, mos_true):
        return torch.nn.functional.mse_loss(mos_pred, mos_true)

    def forward(self, melspec):
        log_melspec = torch.log(melspec + self.eps)
        log_melspec = log_melspec.squeeze(1).permute(0, 2, 1)
        x, _ = self.rnn(log_melspec)
        x = self.fc(x)
        q = self.pool(x)
        q = self.last_activation(q)
        return q.squeeze(-1), x.squeeze(-1)

    def process_data(self, audio):
        # preprocess and run forward pass
        stft = self.preproc(audio)
        mos_pred, framewise_pred = self.forward(stft)
        return (mos_pred, framewise_pred), stft

    def train_step(self, x_input, mos_true):
        (mos_pred, framewise_pred), _ = self.process_data(x_input)
        #loss = self.loss_function(mos_pred, mos_true, framewise_pred=framewise_pred)
        loss = self.loss_function(mos_pred, mos_true)
        return loss
    
    def valid_step(self, x_input, mos_true):
        with torch.no_grad():
            (mos_pred, framewise_pred), _ = self.process_data(x_input)
            loss = self.loss_function(mos_pred, mos_true)
            #loss = self.loss_function(mos_pred, mos_true, framewise_pred=framewise_pred)
        return mos_pred, mos_true, loss
    
    def test_step(self, x_input, mos_true):
        with torch.no_grad():
            (mos_pred, _), _ = self.process_data(x_input)
        return mos_pred, mos_true
    
    def _pick_last_activation(self, last_activation):
        if last_activation == 'none':
            return nn.Identity()
        elif last_activation == 'sigmoid_pesq':
            return  ScoreSigmoid(a_min=-0.5, a_max=4.5, beta=1)
        else:
            raise ValueError('Value of last_activation not valid.')
        
    
    def _initialize_forget_gate_bias(self, hidden_size, bidirectional, forget_bias=-3.0):
        for names in self.rnn.named_parameters():
            if "bias_ih_l0" in names[0]:
                input_bias = names[1].data
                input_bias[hidden_size:2*hidden_size].fill_(forget_bias)
                
            if "bias_hh_l0" in names[0]:
                hidden_bias = names[1].data
                hidden_bias[hidden_size:2*hidden_size].fill_(forget_bias)
                
            if bidirectional:
                if "bias_ih_l0_reverse" in names[0]:
                    input_bias = names[1].data
                    input_bias[hidden_size:2*hidden_size].fill_(forget_bias)
                    
                if "bias_hh_l0_reverse" in names[0]:
                    hidden_bias = names[1].data
                    hidden_bias[hidden_size:2*hidden_size].fill_(forget_bias)
