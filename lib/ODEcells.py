import torch
import math
import numpy as np
from torchdiffeq import odeint

from torch.nn.utils.rnn import pack_padded_sequence
from torch import nn

import lib.utils as utils

# GRU-ODE: Neural Negative Feedback ODE with Bayesian jumps

class GRUODECell(torch.nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        """
        For p(t) modelling input_size should be 2x the x size.
        """
        super().__init__()
        self.input_size  = input_size
        self.hidden_size = hidden_size
        self.bias        = bias

        self.lin_xz = torch.nn.Linear(input_size, hidden_size, bias=bias)
        self.lin_xn = torch.nn.Linear(input_size, hidden_size, bias=bias)

        self.lin_hz = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.lin_hn = torch.nn.Linear(hidden_size, hidden_size, bias=False)


    def forward(self, x, h):
        """
        Returns a change due to one step of using GRU-ODE for all h.
        The step size is given by delta_t.
        Args:
            x        input values
            h        hidden state (current)
            delta_t  time step
        Returns:
            Updated h
        """
        z = torch.sigmoid(self.lin_xz(x) + self.lin_hz(h))
        n = torch.tanh(self.lin_xn(x) + self.lin_hn(z * h))

        dh = (1 - z) * (n - h)
        return dh

class GRUODECell_Autonomous(torch.nn.Module):
    def __init__(self, hidden_size, bias=True):
        """
        For p(t) modelling input_size should be 2x the x size.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.bias        = bias

        #self.lin_xz = torch.nn.Linear(input_size, hidden_size, bias=bias)
        #self.lin_xn = torch.nn.Linear(input_size, hidden_size, bias=bias)

        self.lin_hz = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.lin_hn = torch.nn.Linear(hidden_size, hidden_size, bias=False)


    def forward(self, t, h):
        """
        Returns a change due to one step of using GRU-ODE for all h.
        The step size is given by delta_t.
        Args:
            t        time
            h        hidden state (current)
        Returns:
            Updated h
        """
        x = torch.zeros_like(h)
        z = torch.sigmoid(x + self.lin_hz(h))
        n = torch.tanh(x + self.lin_hn(z * h))

        dh = (1 - z) * (n - h)
        return dh


class FullSTARODECell_Autonomous(torch.nn.Module):
    
    def __init__(self, hidden_size, bias=True):
        """
        For p(t) modelling input_size should be 2x the x size.
        """
        super().__init__()

        #self.lin_xh = torch.nn.Linear(input_size, hidden_size, bias=bias)
        #self.lin_xz = torch.nn.Linear(input_size, hidden_size, bias=bias)
        #self.lin_xr = torch.nn.Linear(input_size, hidden_size, bias=bias)

        #self.lin_x = torch.nn.Linear(input_size, hidden_size * 3, bias=bias)

        self.lin_hh = torch.nn.Linear(hidden_size, hidden_size, bias=True)
        self.lin_hz = torch.nn.Linear(hidden_size, hidden_size, bias=True)
        self.lin_hr = torch.nn.Linear(hidden_size, hidden_size, bias=True)
        self.lin_hk = torch.nn.Linear(hidden_size, 1, bias=True)

    def forward(self, h):
        """
        Executes one step with autonomous GRU-ODE for all h.
        The step size is given by delta_t.
        Args:
            t        time of evaluation
            h        hidden state (current)
        Returns:
            Updated h
        """ 
        #slim

        z = torch.tanh( self.lin_hh(h) )
        k = torch.sigmoid( self.lin_hk(h) )

        dh = torch.tanh( (1-k)*h + k*z) - h

        return dh



class FullSTARODECell_Autonomous_2(torch.nn.Module):
    
    def __init__(self, hidden_size, n_units,
		x_K=None, x_z=None ,h_K=None,
        bias=True):
        """
        For p(t) modelling input_size should be 2x the x size.
        """
        super().__init__()

        if x_K is None:  
            self.x_K = nn.Sequential(
                nn.Linear(hidden_size*2, n_units),
                nn.Tanh(),
                nn.Linear(n_units, hidden_size))
            utils.init_network_weights(self.x_K, initype="ortho")
        else:
            self.x_K = x_K

        if x_z is None:  
            self.x_z = nn.Sequential(
                nn.Linear(hidden_size*2, n_units),
                nn.Tanh(),
                nn.Linear(n_units, hidden_size))
            utils.init_network_weights(self.x_z, initype="ortho")
        else:
            self.x_z = x_z

        if h_K is None: 
            self.h_K = nn.Sequential(
                nn.Linear(hidden_size, n_units),
                nn.Tanh(),
                nn.Linear(n_units, hidden_size))
            utils.init_network_weights(self.h_K, initype="ortho")
        else:
            self.h_K = h_K

    def forward(self, h):
        """
        Executes one step with autonomous GRU-ODE for all h.
        The step size is given by delta_t.
        Args:
            t        time of evaluation
            h        hidden state (current)
        Returns:
            Updated h
        """ 

        # x = torch.cat([torch.zeros_like(h), torch.zeros_like(h)],2) 

        # gate_x_K = self.x_K(x)             # return size torch.Size([1, batch_size, latent_dim])
        # gate_x_z = self.x_z(x)             # return size torch.Size([1, batch_size, latent_dim])
        # gate_h_K = self.h_K(h)        # return size torch.Size([1, batch_size, latent_dim])
        
        # gate_x_K = gate_x_K.squeeze()
        # gate_x_z = gate_x_z.squeeze()
        # gate_h_K = gate_h_K.squeeze()

        # K_gain = torch.sigmoid(gate_x_K + gate_h_K)
        # z = torch.tanh(gate_x_z)
		
        # # dh = K_gain * ( z - h)  
        # dh = torch.tanh( (1-K_gain)*h + K_gain*z) - h
        
        # x = torch.zeros_like(h) 

        # gate_h_K = self.h_K(h)        # return size torch.Size([1, batch_size, latent_dim])
        # gate_h_K = gate_h_K.squeeze()

        # K_gain = torch.sigmoid(x + gate_h_K)
        # dh = torch.tanh( (1-K_gain)*h + x) - h
		

        gate_h_K = self.h_K(h)        # return size torch.Size([1, batch_size, latent_dim]) 
        K_gain = torch.sigmoid(gate_h_K.squeeze())
        dh = torch.tanh( (1-K_gain)*h ) - h

        return dh 



class FullGRUODECell_Autonomous(torch.nn.Module):
    
    def __init__(self, hidden_size, bias=True):
        """
        For p(t) modelling input_size should be 2x the x size.
        """
        super().__init__()

        #self.lin_xh = torch.nn.Linear(input_size, hidden_size, bias=bias)
        #self.lin_xz = torch.nn.Linear(input_size, hidden_size, bias=bias)
        #self.lin_xr = torch.nn.Linear(input_size, hidden_size, bias=bias)

        #self.lin_x = torch.nn.Linear(input_size, hidden_size * 3, bias=bias)

        self.lin_hh = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.lin_hz = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.lin_hr = torch.nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, h):
        """
        Executes one step with autonomous GRU-ODE for all h.
        The step size is given by delta_t.
        Args:
            t        time of evaluation
            h        hidden state (current)
        Returns:
            Updated h
        """
        #xr, xz, xh = torch.chunk(self.lin_x(x), 3, dim=1)
        x = torch.zeros_like(h)
        r = torch.sigmoid(x + self.lin_hr(h))
        z = torch.sigmoid(x + self.lin_hz(h))
        u = torch.tanh(x + self.lin_hh(r * h))

        dh = (1 - z) * (u - h)
        return dh


def init_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.05)

def compute_KL_loss(p_obs, X_obs, M_obs, obs_noise_std=1e-2, logvar=True):
    obs_noise_std = torch.tensor(obs_noise_std)
    if logvar:
        mean, var = torch.chunk(p_obs, 2, dim=1)
        std = torch.exp(0.5*var)
    else:
        mean, var = torch.chunk(p_obs, 2, dim=1)
        ## making var non-negative and also non-zero (by adding a small value)
        std       = torch.pow(torch.abs(var) + 1e-5,0.5)

    return (gaussian_KL(mu_1 = mean, mu_2 = X_obs, sigma_1 = std, sigma_2 = obs_noise_std)*M_obs).sum()


def gaussian_KL(mu_1, mu_2, sigma_1, sigma_2):
    return(torch.log(sigma_2) - torch.log(sigma_1) + (torch.pow(sigma_1,2)+torch.pow((mu_1 - mu_2),2)) / (2*sigma_2**2) - 0.5)
