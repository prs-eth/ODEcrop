import torch
import torch.nn
import torch.nn.functional as F

class model_GT(torch.nn.Module):
    def __init__(self, nclasses=36, nclasses_local_1=None, nclasses_local_2=None, s1_2_s3=None, s2_2_s3=None):
        super(model_GT, self).__init__()
        self.nclasses = nclasses
        self.nclasses_local_1 = nclasses_local_1
        self.nclasses_local_2 = nclasses_local_2
        
        self.hidden_dim = 2 * nclasses
        
        self.s1_2_s3 = s1_2_s3
        self.s2_2_s3 = s2_2_s3

#        self.rnn = torch.nn.RNNCell(nclasses, self.hidden_dim, nonlinearity='relu')
#        self.rnn2 = torch.nn.RNNCell(self.hidden_dim, self.hidden_dim, nonlinearity='relu')

        self.rnn = torch.nn.GRUCell(nclasses, self.hidden_dim)
        self.rnn2 = torch.nn.GRUCell(self.hidden_dim, self.hidden_dim)

        #self.rnn.weight_ih = torch.eye(self.hidden_dim,self.hidden_dim)
        #torch.nn.init.orthogonal_(self.rnn.weight_ih)
        #torch.nn.init.orthogonal_(self.rnn.weight_hh)
        
        self.dense = torch.nn.Linear(self.hidden_dim, nclasses)
        self.activation = torch.nn.ReLU()

        self.dense_l1 = torch.nn.Linear(self.hidden_dim, nclasses_local_1)
        self.activation_l1 = torch.nn.ReLU()

        self.dense_l2 = torch.nn.Linear(self.hidden_dim, nclasses_local_2)
        self.activation_l2 = torch.nn.ReLU()
        
    def forward(self, x, hidden=None):
        x1_, x2_, x3 = x
                
        x1_ = x1_.permute(0,2,3,1)
        x2_ = x2_.permute(0,2,3,1)
        x3 = x3.permute(0,2,3,1)
        
        x1_ = x1_.contiguous().view(-1,x1_.shape[3])
        x2_ = x2_.contiguous().view(-1,x2_.shape[3])
        x3 = x3.contiguous().view(-1,x3.shape[3])

        x1 = torch.zeros_like(x3)
        x2 = torch.zeros_like(x3)
    
        b, c = x3.shape

        if self.s1_2_s3[0] == None:        
            x1[:, :x1_.shape[1]] = x1_[:,:]
            x2[:, :x2_.shape[1]] = x2_[:,:]
        else:
            for i in range(self.s1_2_s3.shape[0]):
                x1[:,i] = x1_[:,int(self.s1_2_s3[i])]
                x2[:,i] = x2_[:,int(self.s2_2_s3[i])]
        
        if hidden is None:
            hidden = torch.zeros((b, self.hidden_dim))
            
        if torch.cuda.is_available():
            hidden = hidden.cuda()

#        for iter in range(t):
#            hidden = self.rnn.forward( x[:,:,iter], hidden )
            
        hidden1 = self.rnn.forward( x1, hidden)
        hidden2 = self.rnn.forward( x2, hidden1)
        hidden3 = self.rnn.forward( x3, hidden2)
        

        hidden = self.rnn2.forward( hidden1, hidden)
        last_l1 = self.dense_l1(hidden)
        last_l1 = self.activation_l1(last_l1)

        hidden = self.rnn2.forward( hidden2, hidden)
        last_l2 = self.dense_l2(hidden)
        last_l2 = self.activation_l2(last_l2)

        hidden = self.rnn2.forward( hidden3, hidden)        
        
        last = self.dense(hidden)
        
        #last = last +  x3   
        last = self.activation(last)
 

        return F.log_softmax(last, dim=1), F.log_softmax(last_l1, dim=1), F.log_softmax(last_l2, dim=1)
