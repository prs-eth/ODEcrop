import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from gpytorch.mlls import ExactMarginalLogLikelihood

train_X = torch.rand(10, 2); print(train_X)
Y = 1 - (train_X - 0.5).norm(dim=-1, keepdim=True); print(Y)  # explicit output dimension
Y += 0.1 * torch.rand_like(Y); print(Y)
train_Y = (Y - Y.mean()) / Y.std(); print(train_Y)

gp = SingleTaskGP(train_X, train_Y)
mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
fit_gpytorch_model(mll)

from botorch.acquisition import UpperConfidenceBound

UCB = UpperConfidenceBound(gp, beta=0.1)

from botorch.optim import optimize_acqf

bounds = torch.stack([torch.zeros(2), torch.ones(2)])
candidate, acq_value = optimize_acqf(
    UCB, bounds=bounds, q=1, num_restarts=5, raw_samples=20,
)



