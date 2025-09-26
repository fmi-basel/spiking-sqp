import torch

import stork
from stork.activations import SuperSpike
from stork.connections import Connection, IdentityConnection
from stork.initializers import FluctuationDrivenCenteredNormalInitializer, DistInitializer
from stork.layers import ConvLayer, Layer
from stork.nodes import (InputGroup, FanOutGroup, MaxPool1d)
from stork.regularizers import UpperBoundL2, UpperBoundL1, LowerBoundL2

from sqp_snn.stork_ext.models import CustomRecurrentSpikingModel
from sqp_snn.stork_ext.lif import CustomLIFGroup
from sqp_snn.stork_ext.adaptive_learn import CustomAdaptLearnLIFGroup
from sqp_snn.stork_ext.readout import CustomReadoutGroup


class SpikingCNN(CustomRecurrentSpikingModel):
    def __init__(self, args, sim_timestep, device, dtype):
        super().__init__(args.batch_size, args.nb_simsteps, args.nb_freqbands, device, dtype)
        
        self.sim_timestep = sim_timestep
        self.device = device
        self.dtype = dtype
        
        act_fn = SuperSpike
        act_fn.beta = args.beta
        
        base_params = dict(
            tau_mem = args.tau_mem_enc,
            activation = act_fn,
            stateful = args.stateful,
            dropout_p = 0.0,
            learn_tau_mem = args.learn_tau_mem,
            hetero_tau_mem = args.hetero_tau_mem
        )
        
        base_params.update({
            'tau_syn': args.tau_syn_enc,
            'learn_tau_syn': args.learn_tau_syn,
            'hetero_tau_syn': args.hetero_tau_syn
        })
            
        self.neuronal_params = base_params
        self.neuron_group = CustomLIFGroup
        
        self.nb_classes = 1
        self.nb_hidden_channels = args.nb_hidden_channels
        self.nu = args.nu
                
        self.initializer = FluctuationDrivenCenteredNormalInitializer(sigma_u = args.sigma_u,
                                                                      nu = self.nu,
                                                                      timestep = self.sim_timestep,
                                                                      alpha = 0.9)
        
        self.readout_initializer = DistInitializer(dist = torch.distributions.Normal(0, 1),
                                                   scaling = '1/sqrt(k)')        
        
        scale_act_reg = 1.0/(args.nb_conv + args.nb_dense + 1.0)

        pop_ubl2 = UpperBoundL2(scale_act_reg*args.pop_ubl2_str, threshold=args.pop_ubl2_thr, dims=(-2, -1))
        pop_ubl1 = UpperBoundL1(scale_act_reg*args.pop_ubl1_str, threshold=args.pop_ubl1_thr, dims=(-2, -1))
        neur_lbl2 = LowerBoundL2(scale_act_reg*args.neur_lbl2_str, threshold=args.neur_lbl2_thr, dims=(-1))                          
        neur_ubl2 = UpperBoundL2(args.neur_ubl2_str, threshold=args.neur_ubl2_thr, dims=False)

        self.regs = [pop_ubl2,
                     pop_ubl1,
                     neur_lbl2,
                     neur_ubl2]
        
        self.plot_groups = []
        self.input_group = self.create_input_group(args)
        self.encoder_group = self.create_encoder_layer(args, self.input_group)
        self.hidden_output = self.create_conv_layers(args, self.encoder_group)
        if args.nb_dense > 0:
            self.hidden_output = self.create_dense_layers(args, self.hidden_output)
        self.readout_group = self.create_readout_layer(args, self.hidden_output)

    def create_input_group(self, args):
        input_shape = (1, args.nb_freqbands)
        input_group = self.add_group(InputGroup(input_shape))
        return input_group

    def create_encoder_layer(self, args, upstream_group):
        fanout_group =  self.add_group(FanOutGroup(upstream_group, args.encoder_fanout, dim=0))
        
        self.neuronal_params['tau_mem'] = args.tau_mem_enc
        self.neuronal_params['tau_syn'] = args.tau_syn_enc
        analog_digital_conversion_group = self.add_group(CustomAdaptLearnLIFGroup(fanout_group.shape,
                                                                                  tau_ada = 100e-3,
                                                                                  name = 'Encoder',
                                                                                  regularizers = self.regs,
                                                                                  **self.neuronal_params))
        
        id_con = self.add_connection(IdentityConnection(fanout_group,
                                                        analog_digital_conversion_group, 
                                                        bias = True,
                                                        tie_weights = [1],
                                                        weight_scale = 1.0))

        # Encoder initialization
        scale = args.encoder_gain * self.sim_timestep/(args.tau_syn_enc)
        shape = id_con.weights.shape
        id_con.weights.data = torch.reshape(torch.linspace(-scale, scale,
                                                           analog_digital_conversion_group.shape[0],
                                                           requires_grad=True), shape)

        self.plot_groups.append(analog_digital_conversion_group)
        return analog_digital_conversion_group

    def create_conv_layers(self, args, upstream_group):
        self.neuronal_params['tau_mem'] = args.tau_mem_hidden
        self.neuronal_params['tau_syn'] = args.tau_syn_hidden
        self.neuronal_params['dropout_p'] = args.dropout_conv
        
        recurrent_params = {
            'kernel_size': args.kernel_size,
            'padding': args.padding
        }
        
        for conv_idx in range(args.nb_conv):            
            conv = ConvLayer(
                name = 'Conv' + str(conv_idx+1), 
                model = self,
                input_group = upstream_group,
                kernel_size = args.kernel_size,
                stride = args.stride,
                padding = args.kernel_size // 2,
                nb_filters = self.nb_hidden_channels,
                recurrent = args.recurrence,
                neuron_class = self.neuron_group,
                neuron_kwargs = self.neuronal_params,
                regs = self.regs,
                recurrent_connection_kwargs = recurrent_params
            )
            
            self.initializer.initialize(conv)
            upstream_group = conv.output_group
            self.plot_groups.append(conv.output_group)            
            
            if args.stride == 1:
                maxpool = self.add_group(MaxPool1d(upstream_group))
                upstream_group = maxpool
            if args.block_depth:
                if (conv_idx+1) % args.block_depth == 0:
                    self.nb_hidden_channels = args.channel_fanout * self.nb_hidden_channels
            elif args.increasing_depth:
                self.nb_hidden_channels = args.channel_fanout * self.nb_hidden_channels
            if self.nb_hidden_channels > args.max_depth:
                self.nb_hidden_channels = args.max_depth
                
            self.nu = self.nu * (1-args.dropout_conv)
            print(f"Conv{conv_idx+1} nu = {self.nu}")
            self.initializer = FluctuationDrivenCenteredNormalInitializer(sigma_u = args.sigma_u,
                                                                          nu = self.nu,
                                                                          timestep = self.sim_timestep,
                                                                          alpha = 0.9,
                                                                          dtype=self.dtype)
                
        return upstream_group
    
    def create_dense_layers(self, args, upstream_group):
        
        self.neuronal_params['dropout_p'] = args.dropout_dense
        
        if args.global_maxpool:
            maxpool = self.add_group(MaxPool1d(upstream_group, kernel_size=upstream_group.shape[-1]))
            upstream_group = maxpool
        
        for dense_idx in range(args.nb_dense):
            self.neuronal_params['tau_mem'] = args.tau_mem_hidden
            self.neuronal_params['tau_syn'] = args.tau_syn_hidden
            
            dense = Layer(name = 'Dense' + str(dense_idx+1), 
                          model = self,
                          size = upstream_group.shape[0],
                          input_group = upstream_group,
                          recurrent = args.recurrence,
                          regs = self.regs,
                          flatten_input_layer = True,
                          neuron_class = self.neuron_group,
                          neuron_kwargs = self.neuronal_params)
            
            self.initializer.initialize(dense)
            upstream_group = dense.output_group
            self.plot_groups.append(dense.output_group)
            
            
            self.nu = self.nu * (1-args.dropout_dense)
            print("Dense" + str(dense_idx+1) + " nu:", self.nu)
            self.initializer = FluctuationDrivenCenteredNormalInitializer(sigma_u = args.sigma_u,
                                                                          nu = self.nu,
                                                                          timestep = self.sim_timestep,
                                                                          alpha = 0.9)
            
        return upstream_group
    
    def create_readout_layer(self, args, upstream_group):
        readout_group = self.add_group(CustomReadoutGroup(self.nb_classes,
                                                          tau_mem = args.tau_mem_readout,
                                                          tau_syn = args.tau_syn_readout,
                                                          learn_tau_mem = args.learn_tau_mem_out,
                                                          hetero_tau_mem = args.hetero_tau_mem_out,
                                                          stateful = args.stateful,
                                                          initial_state = -1e-3))
        
        readout_connection = self.add_connection(Connection(upstream_group, 
                                                            readout_group,
                                                            flatten_input = True))
        self.readout_initializer.initialize(readout_connection)
        return readout_group
