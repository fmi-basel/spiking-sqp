import torch
import math
from torch import nn
from torch.nn import init
from torch.nn.common_types import _size_1_t, _size_2_t, _size_3_t
from typing import Optional, List, Tuple, Union
from torch.nn.modules.utils import _pair
   

class SuperSpike(torch.autograd.Function):
    """
    Autograd SuperSpike nonlinearity implementation.
    The steepness parameter beta can be accessed via the static member
    self.beta.
    """
    beta = 1.0

    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the step function output. ctx is the context object
        that is used to stash information for backward pass computations. 
        """
        ctx.save_for_backward(input)
        out = torch.zeros_like(input)
        out[input > 0] = 1.0
        return out

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the
        loss with respect to the output, and we compute the surrogate gradient
        of the loss with respect to the input. Here we assume the standardized
        negative part of a fast sigmoid as this was done in Zenke & Ganguli
        (2018).
        """
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = grad_input/(SuperSpike.beta*torch.abs(input)+1.0)**2
        return grad


class HeavisideSG(nn.Module):
    r"""Applies the element-wise function with fast sigmoid surrogate gradient:

    .. math::
        \text{Sigmoid}(x) = \Theta(x) 

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    Examples::

        >>> m = nn.HeavisideSG()
        >>> input = torch.randn(2)
        >>> output = m(input)
    """

    def __init__(self, beta=1):
        super(HeavisideSG, self).__init__()
        self.heaviside = SuperSpike.apply
        SuperSpike.beta = beta

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.heaviside(input)


class HeavisideSG_ZeroMean(nn.Module):
    r"""Applies the element-wise Heaviside function with fast sigmoid surrogate gradient and plus-one-minus-one zero mean output:

    .. math::
        \text{Sigmoid}(x) = \Theta(x) 

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.
    """

    def __init__(self, beta=1):
        super(HeavisideSG_ZeroMean, self).__init__()
        self.heaviside = SuperSpike.apply
        SuperSpike.beta = beta

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return 2.0*(self.heaviside(input)-0.5)


class Conv2dBinary(nn.Conv2d):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: Union[str, _size_2_t] = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',  # TODO: refine this type
        device=None,
        dtype=None
    ) -> None:
        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            groups, bias, padding_mode, device, dtype)

        # Overwrite standard init
        init.uniform_(self.weight, -0.05, 0.05)
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
        self.scl = math.sqrt(2.0/fan_in)

        if self.bias is not None:
            init.zeros_(self.bias)
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        discrete_weight = 2*(SuperSpike.apply(self.weight)-0.5)
        return self._conv_forward(input, discrete_weight, self.bias)*self.scl
        # test_weight = torch.tanh(self.weight)
        # test_weight = self.weight
        # return self._conv_forward(input, test_weight, self.bias)*self.scl


def get_Conv2dBlock(in_channels, out_channels, dropout=None, last=False, gap=False, activation=nn.ReLU(), weight_quant=None):
    block = []
    if weight_quant == 'binary':
        block.append(Conv2dBinary(in_channels, out_channels, kernel_size=3, padding="same"))
    else:
        block.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding="same"))
    block.append(activation)
    if not last:
        block.append(nn.MaxPool2d(kernel_size=(2, 2)))
    else:
        if gap:
            block.append(nn.AdaptiveAvgPool2d(1))
            block.append(nn.Flatten())
        else:
            block.append(nn.AdaptiveMaxPool2d(1))
            block.append(nn.Flatten())
    if dropout is not None:
        block.append(nn.Dropout(dropout))
    return nn.Sequential(*block)


def get_activation_function(name, conv_activation_param):
    """ Translate name of activation function in object reference. """ 
    if type(name) is not str:
        raise TypeError 
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    elif name == "sigmoid":
        return nn.Sigmoid()
    elif name == "heaviside":
        return HeavisideSG(conv_activation_param)
    elif name == "heaviside_zm":
        return HeavisideSG_ZeroMean(conv_activation_param)
    elif name == "linear":
        return None
    else:
        raise TypeError 


class DNSMOS(nn.Module):
    def __init__(self, n_fft, n_mels, channels, hidden_1, hidden_2, dropout, preproc, conv_activation=nn.ReLU(), conv_activation_param=None, gap=False, weight_quant=None):
        super(DNSMOS, self).__init__()
        self.n_fft = n_fft
        self.n_mels = n_mels
        conv_activation = get_activation_function(conv_activation, conv_activation_param)
        # conv blocks
        self.conv1 = get_Conv2dBlock(1, channels[0], dropout, activation=conv_activation, weight_quant=weight_quant)
        self.conv2 = get_Conv2dBlock(channels[0], channels[1], dropout=dropout, activation=conv_activation, weight_quant=weight_quant)
        self.conv3 = get_Conv2dBlock(channels[1], channels[2], dropout=dropout, activation=conv_activation, weight_quant=weight_quant)
        self.conv4 = get_Conv2dBlock(channels[2], channels[3], dropout=None, activation=conv_activation, weight_quant=weight_quant, last=True, gap=gap)
        # dense blocks
        self.dense = nn.Sequential(
            nn.Linear(channels[3], hidden_1),
            nn.ReLU(),
            nn.Linear(hidden_1, hidden_2),
            nn.ReLU(),
            nn.Linear(hidden_2, 1))
        # proc
        self.preproc = preproc
        # other
        self.eps = 1e-9

    def forward(self, melspec_input):
        # log power
        log_melspec_input = torch.log(melspec_input + self.eps)
        # neural network layers 
        c1 = self.conv1(log_melspec_input)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        c4 = self.conv4(c3)
        mos_pred = self.dense(c4)
        return mos_pred

    def loss_function(self, mos_pred, mos_true, melspec_input):
        return torch.nn.functional.mse_loss(mos_pred, mos_true)

    def process_data(self, x_input):
        # preprocess and run forward pass
        melspec_input = self.preproc(x_input)
        mos_pred = self.forward(melspec_input).squeeze(-1)
        return mos_pred, melspec_input

    def train_step(self, x_input, mos_true):
        mos_pred, melspec_input = self.process_data(x_input)
        loss = self.loss_function(mos_pred, mos_true, melspec_input)
        return loss
    
    def valid_step(self, x_input, mos_true):
        with torch.no_grad():
            mos_pred, melspec_input = self.process_data(x_input)
            loss = self.loss_function(mos_pred, mos_true, melspec_input)
        return mos_pred, mos_true, loss
    
    def test_step(self, x_input, mos_true):
        with torch.no_grad():
            mos_pred, _ = self.process_data(x_input)
        return mos_pred, mos_true
