import torch
import torch.nn
from models.convstar.convstar import ConvSTAR

import torch.nn.functional as F

class STARSequentialEncoder(torch.nn.Module):
    def __init__(self, height, width, input_dim=9, hidden_dim=64, nclasses=8, kernel_size=(3,3), n_layers=2, use_in_layer_norm=False):
        super(STARSequentialEncoder, self).__init__()
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        #self.inconv = torch.nn.Conv3d(input_dim,hidden_dim,(1,3,3))
        
        self.use_in_layer_norm = use_in_layer_norm
        if use_in_layer_norm:
            self.in_layer_norm = torch.nn.LayerNorm(input_dim)
        
        self.rnn1 = ConvSTAR(input_size=input_dim, 
                                hidden_sizes=hidden_dim,
                                kernel_sizes=kernel_size[0], 
                                n_layers=n_layers)
        
        self.rnn2 = ConvSTAR(input_size=hidden_dim, 
                                hidden_sizes=hidden_dim,
                                kernel_sizes=kernel_size[0], 
                                n_layers=n_layers)


        self.final_1 = torch.nn.Conv2d(hidden_dim, hidden_dim*2, (3, 3), padding=1)
        self.f_bn_1 = torch.nn.BatchNorm2d(hidden_dim*2)
        self.final_2 = torch.nn.Conv2d(hidden_dim*2, nclasses, (3, 3), padding=1)
        self.f_bn_2 = torch.nn.BatchNorm2d(nclasses)

        self.temporal_max_pooling = torch.nn.MaxPool3d((2,1,1))

    def forward(self, x, hiddenS=None):
        
    
        if self.use_in_layer_norm:
            #(b x t x c x h x w) -> (b x t x h x w x c) -> (b x c x t x h x w)
            x = self.in_layer_norm(x.permute(0,1,3,4,2)).permute(0,4,1,2,3)
        else:
            # (b x t x c x h x w) -> (b x c x t x h x w)
            x = x.permute(0,2,1,3,4)
            
        #x = torch.nn.functional.pad(x, (1, 1, 1, 1), 'constant', 0)
        #x = self.inconv.forward(x)

        b, c, t, h, w = x.shape


        #convRNN step---------------------------------
        #hiddenS is a list (number of layer) of hidden states of size [b x c x h x w]
        if hiddenS is None:
            hiddenS_1 = [torch.zeros((b, self.hidden_dim, h, w))] * self.n_layers
            hiddenS_2 = [torch.zeros((b, self.hidden_dim, h, w))] * self.n_layers
            out_rnn_1 = torch.zeros((b, self.hidden_dim,t, h, w))
            out_rnn_2 = torch.zeros((b, self.hidden_dim,t//2, h, w))
            
        if torch.cuda.is_available():
            for i in range(self.n_layers):
                hiddenS_1[i] = hiddenS_1[i].cuda()
                hiddenS_2[i] = hiddenS_2[i].cuda()
                out_rnn_1 = out_rnn_1.cuda()
                out_rnn_2 = out_rnn_2.cuda()

        for iter in range(t):
            hiddenS_1 = self.rnn1.forward( x[:,:,iter,:,:], hiddenS_1 )
            out_rnn_1[:,:,iter,:,:] = hiddenS_1[-1]

        #Temporal max pooling
        out_rnn_1 = self.temporal_max_pooling(out_rnn_1)


        for iter in range(t//2):
            hiddenS_2 = self.rnn2.forward( out_rnn_1[:,:,iter,:,:], hiddenS_2 )
            out_rnn_2[:,:,iter,:,:] = hiddenS_2[-1]        

        #Temporal max pooling
        out_rnn_2 = self.temporal_max_pooling(out_rnn_2)


        last = torch.mean(out_rnn_2, dim=2)

        #last = torch.nn.functional.pad(last, (1, 1, 1, 1), 'constant', 0)
        last = self.final_1(last)
        last = F.relu(self.f_bn_1(last))
        last = self.final_2(last)
        last = F.relu(self.f_bn_2(last))

        return F.log_softmax(last, dim=1)


if __name__=="__main__":


    b, t, c, h, w = 2, 10, 3, 320, 320

    model = STARSequentialEncoder(height=h, width=w, input_dim=c, hidden_dim=3)

    x = torch.randn((b, t, c, h, w))


    hidden, state = model.forward(x)


