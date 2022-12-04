import numpy as np 
import math

import sys
sys.path.append('../../')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn.init as init
from utils.myModels.ConvGRU import ConvGRU
from utils.myModels.unet import UNet

class NAOMI(nn.Module):
    def __init__(self, gru_hidden_dim = 128, gru_input_dim = 128, num_layers = 3, device = torch.device('cpu')):
        super(NAOMI, self).__init__()
        self.gru_hidden_dim = gru_hidden_dim
        self.gru_input_dim = gru_input_dim
        self.num_layers = num_layers

        self.gru = ConvGRU(input_size=1, hidden_sizes=[32,64,16],
                  kernel_sizes=[3, 5, 3], n_layers=3, device = device)
        self.back_gru = ConvGRU(input_size=1, hidden_sizes=[32,64,16],
                  kernel_sizes=[3, 5, 3], n_layers=3, device = device)
        
        self.combiner = UNet(n_channels = 32, n_classes = 1, bilinear=False)
        # hardcode 5 as input length


    # take in a sequence of encodings as input along with holes
    def forward(self, seq):
        # hardcode 5 as input length
        # input = 100, 5, 64, 64
        # needed =  100, 1, 64, 64
        seq_ans = torch.clone(seq)
        assert(seq.shape[1] == 5)
        hidden_forward = [None for i in range(5)]
        hidden_backward = [None for i in range(5)]
        hidden_forward[0] = self.gru(seq[:,0:1,:,:])
        hidden_backward[4] = self.back_gru(seq[:,4:5,:,:])
        # Got 100, 16chan, 64, 64
        # Needed 100, 32, 64, 64
        # outp 100, 1, 64, 64

        seq_ans[:,2,:] = self.combiner(torch.cat((hidden_forward[0][-1], hidden_backward[4][-1]), 1)).squeeze()
        hidden_forward[2] = self.gru(seq_ans[:,2:3,:], hidden_forward[0])
        hidden_backward[2] = self.back_gru(seq_ans[:,2:3,:], hidden_backward[4])

        seq_ans[:,1,:] = self.combiner(torch.cat((hidden_forward[0][-1], hidden_backward[2][-1]), 1)).squeeze()
        seq_ans[:,3,:] = self.combiner(torch.cat((hidden_forward[2][-1], hidden_backward[4][-1]), 1)).squeeze()

        return seq_ans

