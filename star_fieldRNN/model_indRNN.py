import torch
import torch.nn.functional as F
from torch import nn
from indRNN import IndRNN


class model_indRNN(nn.Module):
    def __init__(self, input_size=55, output_size=23, nhidden=110, nlayers=4, dropout=0.0):
        super(model_indRNN, self).__init__()
        self.h0 = torch.zeros(nlayers, 24, nhidden)
        self.ind = IndRNN(input_size, nhidden, nlayers, nonlinearity='relu',  batch_first=True)
        self.linear = nn.Linear(nhidden, output_size)

    def forward(self, inputs):

        y = self.ind(inputs, self.h0)[0] 
        o = self.linear(y[:,-1,:])
        return F.log_softmax(o, dim=1)
    
    
class model_indRNN_dense(nn.Module):
    def __init__(self, input_size=55, output_size=23, nhidden=110, nlayers=4, dropout=0.0):
        super(model_indRNN_dense, self).__init__()
        self.h0 = torch.zeros(nlayers, 24, nhidden)
        self.ind = IndRNN(input_size, nhidden, nlayers, nonlinearity='relu',  batch_first=True)
        self.linear = nn.Linear(nhidden, output_size)

    def forward(self, inputs):

        y = self.ind(inputs, self.h0)[0] 
        #Reshape
        y = y.view(y.shape[0]*y.shape[1], y.shape[2])
        
        y = self.linear(y)
        return F.log_softmax(y, dim=-1)