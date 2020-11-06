"""
author: Nando Metzger
metzgern@ethz.ch
"""


import torch


class FullGRUODECell_Autonomous(torch.nn.Module):
	""" 
	From Paper "GRU-ODE-Bayes: Continuous modeling of sporadically-observed time series"
	De Brouwer, Simm, Arany, Moreau, NeurIPS 2019
	https://github.com/edebrouwer/gru_ode_bayes
	"""
	def __init__(self, hidden_size, bias=True):
		"""
		For p(t) modelling input_size should be 2x the x size.
		"""
		super().__init__()

		#self.lin_xh = torch.nn.Linear(input_size, hidden_size, bias=bias)
		#self.lin_xz = torch.nn.Linear(input_size, hidden_size, bias=bias)
		#self.lin_xr = torch.nn.Linear(input_size, hidden_size, bias=bias)

		#self.lin_x = torch.nn.Linear(input_size, hidden_size * 3, bias=bias)

		self.lin_hh = torch.nn.Linear(hidden_size, hidden_size, bias=bias)
		self.lin_hz = torch.nn.Linear(hidden_size, hidden_size, bias=bias)
		self.lin_hr = torch.nn.Linear(hidden_size, hidden_size, bias=bias)

	def forward(self, h):
	#def forward(self, t, h):
		"""
		Executes one step with autonomous GRU-ODE for all h.
		The step size is given by delta_t.

		Args:
			t		time of evaluation
			h		hidden state (current)

		Returns:
			Updated h
		"""
		#xr, xz, xh = torch.chunk(self.lin_x(x), 3, dim=1)
		x = torch.zeros_like(h)
		r = torch.sigmoid(x + self.lin_hr(h))
		z = torch.sigmoid(x + self.lin_hz(h))
		g = torch.tanh(x + self.lin_hh(r * h))

		dh = (1 - z) * (g - h)
		return dh

class FullGRUODECell(torch.nn.Module):
	"""
	dFrom Paper "GRU-ODE-Bayes: COntinuous modeling of sporadically-observed time series"
	De Brouwer, Simm, Arany, Moreau, NeurIPS 2019
	https://github.com/edebrouwer/gru_ode_bayes
	"""
	def __init__(self, input_size, hidden_size, bias=True):
		"""
		For p(t) modelling input_size should be 2x the x size.
		"""
		super().__init__()

		#self.lin_xh = torch.nn.Linear(input_size, hidden_size, bias=bias)
		#self.lin_xz = torch.nn.Linear(input_size, hidden_size, bias=bias)
		#self.lin_xr = torch.nn.Linear(input_size, hidden_size, bias=bias)

		self.lin_x = torch.nn.Linear(input_size, hidden_size * 3, bias=bias)

		self.lin_hh = torch.nn.Linear(hidden_size, hidden_size, bias=False)
		self.lin_hz = torch.nn.Linear(hidden_size, hidden_size, bias=False)
		self.lin_hr = torch.nn.Linear(hidden_size, hidden_size, bias=False)

	def forward(self, x, h):
		"""
		Executes one step with GRU-ODE for all h.
		The step size is given by delta_t.

		Args:
			x		input values
			h		hidden state (current)
			delta_t  time step

		Returns:
			Updated h
		"""
		xr, xz, xh = torch.chunk(self.lin_x(x), 3, dim=1)
		r = torch.sigmoid(xr + self.lin_hr(h))
		z = torch.sigmoid(xz + self.lin_hz(h))
		u = torch.tanh(xh + self.lin_hh(r * h))

		dh = (1 - z) * (u - h)
		return dh



