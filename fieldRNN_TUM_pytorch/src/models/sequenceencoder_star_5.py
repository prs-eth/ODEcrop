import torch
import torch.nn
from models.convstar.convstar import ConvSTAR

import torch.nn.functional as F

class STARSequentialEncoder(torch.nn.Module):
    def __init__(self, height, width, input_dim=9, hidden_dim=64, nclasses=8, kernel_size=(3,3), n_layers=2, use_in_layer_norm=False):
        super(STARSequentialEncoder, self).__init__()
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.inconv = torch.nn.Conv3d(input_dim,hidden_dim,(1,3,3) , padding=1)
        
        self.use_in_layer_norm = use_in_layer_norm
        if use_in_layer_norm:
            self.in_layer_norm = torch.nn.LayerNorm(input_dim)
        
        self.rnn = ConvSTAR(input_size=hidden_dim, 
                                hidden_sizes=hidden_dim,
                                kernel_sizes=kernel_size[0], 
                                n_layers=n_layers)



        self.final_1 = torch.nn.Conv2d(hidden_dim, hidden_dim*2, (3, 3), padding=1)
        self.f_bn_1 = torch.nn.BatchNorm2d(hidden_dim*2)
        self.final_2 = torch.nn.Conv2d(hidden_dim*2, nclasses, (3, 3), padding=1)
        self.f_bn_2 = torch.nn.BatchNorm2d(nclasses)

    def forward(self, x, hiddenS=None):
        
    
        if self.use_in_layer_norm:
            #(b x t x c x h x w) -> (b x t x h x w x c) -> (b x c x t x h x w)
            x = self.in_layer_norm(x.permute(0,1,3,4,2)).permute(0,4,1,2,3)
        else:
            # (b x t x c x h x w) -> (b x c x t x h x w)
            x = x.permute(0,2,1,3,4)
            
        #x = torch.nn.functional.pad(x, (1, 1, 1, 1), 'constant', 0)
        x = self.inconv.forward(x)

        b, c, t, h, w = x.shape


        #convRNN step---------------------------------
        #hiddenS is a list (number of layer) of hidden states of size [b x c x h x w]
        if hiddenS is None:
            hiddenS = [torch.zeros((b, self.hidden_dim, h, w))] * self.n_layers
            
        if torch.cuda.is_available():
            for i in range(self.n_layers):
                hiddenS[i] = hiddenS[i].cuda()

        for iter in range(t):
            hiddenS = self.rnn.forward( x[:,:,iter,:,:], hiddenS )
                    
        last = hiddenS[-1]

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


