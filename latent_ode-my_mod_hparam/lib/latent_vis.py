###########################
# Crop Classification under Varying Cloud Coverwith Neural Ordinary Differential Equations
# Author: Nando Metzger
###########################

from sklearn.decomposition import PCA
import torch
from matplotlib import pyplot as plt
import numpy as np

def get_pca_traj(latent_info, num_PCA=10, num_train_PCA=10, PCA_dim=2):

	# Prepare data
	traj = []
	for latent_step in latent_info:
		traj.append(latent_step["ode_sol"].detach()[:,:num_train_PCA])
	traj = torch.cat(traj, dim=2)
	latent_dim = traj.shape[3]

	# Fit pca
	latPCA = PCA(n_components=PCA_dim).fit(traj.squeeze().reshape(-1, latent_dim).cpu())

	# Apply PCA
	PCA_traj = [None]*num_PCA
	for tr in range(num_PCA):
		PCA_traj[tr] = latPCA.transform(traj.squeeze()[tr].cpu())

	return PCA_traj



def get_pca_fig(Trajectories):
	
	samples, dim = Trajectories[0].shape

	fig = plt.figure()
	ax = fig.add_subplot(1, 1, 1)

	for tr in Trajectories:
		
		if dim==1:
			ax.plot(tr[:,0], np.arange(samples), c='r', marker='1')

		elif dim==2:
			ax.plot(tr[:,0], tr[:,1], c='r', marker='1')

		elif dim==3:
			ax.plot(tr[:,0], tr[1], tr[:,2], c='r', marker='1')

	return fig

