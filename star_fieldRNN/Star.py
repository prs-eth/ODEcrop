import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import os
from ClassificationModel import ClassificationModel
from ourRNN import ourRNNModel as star_cell
 
class STAR(ClassificationModel):
    def __init__(self, input_dim=1, hidden_dims=3, nclasses=5, num_rnn_layers=4, dropout=0.0, bidirectional=False,
                 use_batchnorm=False, use_layernorm=False):

        super(STAR, self).__init__()
        print('STAR'*1)
        self.nclasses=nclasses
        self.use_batchnorm = use_batchnorm
        self.use_layernorm = use_layernorm

        self.d_model = num_rnn_layers*hidden_dims

        if use_layernorm:
            self.inlayernorm = nn.LayerNorm(input_dim)
            self.clayernorm = nn.LayerNorm((hidden_dims + hidden_dims * bidirectional) * num_rnn_layers)

        self.rnn1 = star_cell(input_dim=input_dim, hidden_dim=hidden_dims, droput_factor=dropout, batch_norm=False)
        self.rnn2 = star_cell(input_dim=hidden_dims, hidden_dim=hidden_dims, droput_factor=0.0, batch_norm=False)
        self.rnn3 = star_cell(input_dim=hidden_dims, hidden_dim=hidden_dims, droput_factor=0.0, batch_norm=False)
        self.rnn4 = star_cell(input_dim=hidden_dims, hidden_dim=hidden_dims, droput_factor=0.0, batch_norm=False)
        self.rnn5 = star_cell(input_dim=hidden_dims, hidden_dim=hidden_dims, droput_factor=0.0, batch_norm=False)
        self.rnn6 = star_cell(input_dim=hidden_dims, hidden_dim=hidden_dims, droput_factor=0.0, batch_norm=False)
#        self.rnn7 = star_cell(input_dim=hidden_dims, hidden_dim=hidden_dims, droput_factor=0.2)
#        self.rnn8 = star_cell(input_dim=hidden_dims, hidden_dim=hidden_dims, droput_factor=0.2)



        if bidirectional:
            hidden_dims = hidden_dims * 2

        self.linear_class = nn.Linear(hidden_dims * 1  , nclasses, bias=True)

        if use_batchnorm:
            self.bn = nn.BatchNorm1d(hidden_dims * 1)

#        self.polling = torch.nn.MaxPool1d(4)

    def _logits(self, x):

        #x = x.transpose(1,2)

        if self.use_layernorm:
            x = self.inlayernorm(x)

        outputs1, _ = self.rnn1.forward(x)
        outputs2, _ = self.rnn2.forward(outputs1)            
        outputs3, _ = self.rnn3.forward(outputs2)
        outputs4, _ = self.rnn4.forward(outputs3)
        outputs5, _ = self.rnn5.forward(outputs4)
        outputs6, _ = self.rnn6.forward(outputs5)
#        outputs7, _ = self.rnn7.forward(outputs6)
#        outputs8, _ = self.rnn8.forward(outputs7)
      
        outputs = outputs6
        
        if self.use_batchnorm:
            outputs = outputs[:,-1:,:]
            b,t,d = outputs.shape
            o_ = outputs.view(b, -1, d).permute(0,2,1)
            outputs = self.bn(o_).permute(0, 2, 1).view(b,t,d)

        #h=outputs[:,-1,:] 
        
        outputs = outputs.contiguous().view(outputs.shape[0]*outputs.shape[1], outputs.shape[2])
        logits = self.linear_class.forward(outputs)

        return logits

    def forward(self,x):
        logits = self._logits(x)

        logprobabilities = F.log_softmax(logits, dim=-1)
        # stack the lists to new tensor (b,d,t,h,w)
        return logprobabilities

    def save(self, path="model.pth", **kwargs):
        print("\nsaving model to "+path)
        model_state = self.state_dict()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(dict(model_state=model_state,**kwargs),path)

    def load(self, path):
        print("loading model from "+path)
        snapshot = torch.load(path, map_location="cpu")
        model_state = snapshot.pop('model_state', snapshot)
        self.load_state_dict(model_state)
        return snapshot
