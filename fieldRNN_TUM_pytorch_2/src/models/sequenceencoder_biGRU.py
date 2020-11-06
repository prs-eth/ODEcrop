import torch
import torch.nn
from models.convlstm.convlstm import ConvLSTMCell
from models.convgru.convgru import ConvGRUCell

import torch.nn.functional as F

class biGRUSequentialEncoder(torch.nn.Module):
    def __init__(self, height, width, input_dim=9, hidden_dim=64, nclasses=8, kernel_size=(3,3), bias=False):
        super(biGRUSequentialEncoder, self).__init__()
        self.hidden_dim = hidden_dim

        self.cell_1 = ConvGRUCell(input_size = input_dim, 
                                hidden_size = hidden_dim,
                                kernel_size = kernel_size[0])

        self.cell_2 =  ConvGRUCell(input_size = input_dim, 
                                hidden_size =  hidden_dim,
                                kernel_size = kernel_size[0])

        self.final = torch.nn.Conv2d(hidden_dim*2, nclasses, (3, 3), padding=1)
        

    def forward(self, x, hidden1=None, hidden2=None):

        # (b x t x c x h x w) -> (b x c x t x h x w)
        x = x.permute(0,2,1,3,4)

        b, _, t, h, w = x.shape
        c = self.hidden_dim

        if hidden1 is None:
            hidden1 = torch.zeros((b, c, h, w))        
        if hidden2 is None:
            hidden2 = torch.zeros((b, c, h, w))

        if torch.cuda.is_available():
            hidden1 = hidden1.cuda()
            hidden2 = hidden2.cuda()


        for iter in range(t):

            hidden1 = self.cell_1.forward(x[:,:,iter,:,:], (hidden1))
            hidden2 = self.cell_2.forward(x[:,:,-iter,:,:], (hidden2))

        last = torch.cat( (hidden1,hidden2), 1 )
        last = self.final.forward(last)

        return F.log_softmax(last, dim=1)


