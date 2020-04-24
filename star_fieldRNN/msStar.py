import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import os
from ClassificationModel import ClassificationModel
from ourRNN import ourRNNModel as star_cell
 
class msSTAR(ClassificationModel):
    def __init__(self, input_dim=1, hidden_dims=3, nclasses=23, num_rnn_layers=6, dropout=0.2, bidirectional=False,
                 use_batchnorm=True, use_layernorm=False):

        super(msSTAR, self).__init__()
        
        self.nclasses=nclasses
        self.use_batchnorm = use_batchnorm
        self.use_layernorm = use_layernorm

        
        num_rnn_layers = 6
        print('STAR\n'*num_rnn_layers)
        self.d_model = num_rnn_layers*hidden_dims
        
        if use_layernorm:
            self.inlayernorm = nn.LayerNorm(input_dim)
            self.clayernorm = nn.LayerNorm((hidden_dims + hidden_dims * bidirectional) )
            self.clayernorm_local_1 = nn.LayerNorm((hidden_dims + hidden_dims * bidirectional) )
            self.clayernorm_local_2 = nn.LayerNorm((hidden_dims + hidden_dims * bidirectional) )


        self.cell1 = star_cell(input_dim=input_dim, hidden_dim=hidden_dims, droput_factor=dropout)
        self.cell2 = star_cell(input_dim=hidden_dims, hidden_dim=hidden_dims, droput_factor=dropout)
        self.cell3 = star_cell(input_dim=hidden_dims, hidden_dim=hidden_dims, droput_factor=dropout)
        self.cell4 = star_cell(input_dim=hidden_dims, hidden_dim=hidden_dims, droput_factor=dropout)
        self.cell5 = star_cell(input_dim=hidden_dims, hidden_dim=hidden_dims, droput_factor=dropout)
        self.cell6 = star_cell(input_dim=hidden_dims, hidden_dim=hidden_dims, droput_factor=dropout)
        
        if bidirectional:
            hidden_dims = hidden_dims * 2

        self.linear_class = nn.Linear(hidden_dims, nclasses, bias=True)
        self.linear_class_local_1 = nn.Linear(hidden_dims, 5, bias=True)
        self.linear_class_local_2 = nn.Linear(hidden_dims, 8, bias=True)

        if use_batchnorm:
            if bidirectional:
                self.bn = nn.BatchNorm1d(hidden_dims*2)
                self.bn_local_1 = nn.BatchNorm1d(hidden_dims*2)
                self.bn_local_2 = nn.BatchNorm1d(hidden_dims*2)
            else:
                self.bn = nn.BatchNorm1d(hidden_dims)
                self.bn_local_1 = nn.BatchNorm1d(hidden_dims)
                self.bn_local_2 = nn.BatchNorm1d(hidden_dims)


    def _logits(self, x):
        x = x.transpose(1,2)

        if self.use_layernorm:
            x = self.inlayernorm(x)

        outputs1, _ = self.cell1.forward(x)
        outputs2, _ = self.cell2.forward(outputs1)            
        outputs3, _ = self.cell3.forward(outputs2)
        outputs4, _ = self.cell4.forward(outputs3)
        outputs5, _ = self.cell5.forward(outputs4)
        outputs, _ = self.cell6.forward(outputs5)

        outputs_local1 = outputs2
        outputs_local2 = outputs4        
        
        if self.use_batchnorm:
            outputs = outputs[:,-1:,:]
            b,t,d = outputs.shape
            outputs = self.bn(outputs.view(b, -1, d).permute(0,2,1)).permute(0, 2, 1).view(b,t,d)

            outputs_local1 = outputs_local1[:,-1:,:]
            b,t,d = outputs_local1.shape
            outputs_local1 = self.bn_local_1(outputs_local1.view(b, -1, d).permute(0,2,1)).permute(0, 2, 1).view(b,t,d)

            outputs_local2 = outputs_local2[:,-1:,:]
            b,t,d = outputs_local2.shape
            outputs_local2 = self.bn_local_2(outputs_local2.view(b, -1, d).permute(0,2,1)).permute(0, 2, 1).view(b,t,d)

        outputs=outputs[:,-1,:] 
        outputs_local1 = outputs_local1[:,-1,:] 
        outputs_local2 = outputs_local2[:,-1,:] 
        
        if self.use_layernorm:
            outputs = self.clayernorm(outputs)
            outputs_local1 = self.clayernorm_local_1(outputs_local1)
            outputs_local2 = self.clayernorm_local_2(outputs_local2)

        logits = self.linear_class.forward(outputs)
        logits_local_1 = self.linear_class_local_1.forward(outputs_local1)
        logits_local_2 = self.linear_class_local_2.forward(outputs_local2)

        return logits, logits_local_1, logits_local_2

    def forward(self,x):
        logits, logits_local_1, logits_local_2 = self._logits(x)

        logprobabilities = F.log_softmax(logits, dim=-1)
        logprobabilities_local_1 = F.log_softmax(logits_local_1, dim=-1)
        logprobabilities_local_2 = F.log_softmax(logits_local_2, dim=-1)

        # stack the lists to new tensor (b,d,t,h,w)
        return logprobabilities, logprobabilities_local_1, logprobabilities_local_2

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
