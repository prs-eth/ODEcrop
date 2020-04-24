import torch
import torch.nn as nn
from torch.autograd import Variable
#from torch.nn import Parameter
#from torch import Tensor
import torch.nn.functional as F
import torch.nn.init as init
import math
import numpy as np

class ourRNNCell(nn.Module):

    def __init__(self, input_size, hidden_size, bias=True):
        super(ourRNNCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        
        self.x_K = nn.Linear(input_size,  hidden_size, bias=bias)
        self.x_z = nn.Linear(input_size,  hidden_size, bias=bias)
        self.h_K = nn.Linear(hidden_size,  hidden_size, bias=bias)
        
        init.orthogonal_(self.x_K.weight) 
        init.orthogonal_(self.x_z.weight)
        init.orthogonal_(self.h_K.weight)
        
#        init.kaiming_normal_(self.x_K.weight) 
#        init.kaiming_normal_(self.x_z.weight)
#        init.kaiming_normal_(self.h_K.weight)

        #bias_f= np.log(np.random.uniform(1,784,hidden_size))
        #bias_f = torch.Tensor(bias_f)  
        #self.bias_K = Variable(bias_f.cuda(), requires_grad=True)

        self.x_K.bias.data.fill_(0.)
        self.x_z.bias.data.fill_(0)
        
    def forward(self, x, hidden):
        
        x = x.view(-1, x.size(1))
        
        gate_x_K = self.x_K(x) 
        gate_x_z = self.x_z(x) 
        gate_h_K = self.h_K(hidden)
        
        gate_x_K = gate_x_K.squeeze()
        gate_x_z = gate_x_z.squeeze()
        gate_h_K = gate_h_K.squeeze()
        

        K_gain = torch.sigmoid(gate_x_K + gate_h_K)
        z = torch.tanh(gate_x_z)
        
        #h_new = K_gain * hidden + (1 - K_gain) * z 
        h_new = hidden + K_gain * ( z - hidden) 
        h_new = torch.tanh(h_new)
        
        return h_new

class ourRNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, bias=True, droput_factor=0.0, batch_norm=False):
        super(ourRNNModel, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim
        self.cell = ourRNNCell(input_dim, hidden_dim, bias)
        self.droput_factor = droput_factor
        self.batch_norm = batch_norm
        
        if self.droput_factor != 0:
            self.naive_dropout = nn.Dropout(p=droput_factor)

        if batch_norm:
            print('batch norm')
            self.bn_layer = nn.BatchNorm1d(hidden_dim)
        
        
    def forward(self, x):
        
        # Initialize hidden state with zeros
        if torch.cuda.is_available():
            h0 = Variable(torch.zeros(x.size(0), self.hidden_dim).cuda())
            outs = Variable(torch.zeros(x.size(0), x.shape[1], self.hidden_dim).cuda())
        else:
            h0 = Variable(torch.zeros(x.size(0),  self.hidden_dim))
            outs = Variable(torch.zeros(x.size(0), x.shape[1], self.hidden_dim))
       
        hn = h0
        
        for seq in range(x.size(1)):
            hn = self.cell(x[:,seq], hn) 
        
            if self.droput_factor != 0:
                outs[:,seq,:] = self.naive_dropout(hn)
            else:
                outs[:,seq,:] = hn
             
        #batch normalization:
        if self.batch_norm:
            outs = self.bn_layer(outs.permute(0,2,1)).permute(0,2,1) 
        
        return outs, outs