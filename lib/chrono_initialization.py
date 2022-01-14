import numbers

import torch


def chrono_init(tensor, Tmax = 784, Tmin=1):
    '''chrono initialization(Ref: https://arxiv.org/abs/1804.11188)
    '''
    assert isinstance(Tmin, numbers.Number), 'Tmin must be numeric.'
    assert isinstance(Tmax, numbers.Number), 'Tmax must be numeric.'   
    num_units = tensor.data.shape[1]
    tensor.data = torch.log(torch.nn.init.uniform_(tensor.data, 1, Tmax - 1))


