
# Author: Nando Metzger


from sklearn.decomposition import PCA
import torch
from matplotlib import pyplot as plt
import numpy as np

<<<<<<< HEAD
def get_pca_traj(latent_info, num_PCA=10, num_train_PCA=10, PCA_dim=2):

	# Prepare data
	traj = []
	for latent_step in latent_info:
		traj.append(latent_step["ode_sol"].detach()[:,:num_train_PCA])
	traj = torch.cat(traj, dim=2)
	latent_dim = traj.shape[3]

	# Fit pca
	latPCA = PCA(n_components=PCA_dim).fit(traj.squeeze().reshape(-1, latent_dim).cpu())
=======
def get_pca_traj(latent_info, PCA_dim=1):

	num_PCA = latent_info[0]["ode_sol"].shape[1]

	# Prepare data
	traj = []
	tps = []
	Marker = []
	for latent_step in latent_info:
		traj.append(latent_step["ode_sol"].detach())

		time_points = latent_step["time_points"]
		tps.append(time_points)

		Marker.append(latent_step["marker"])
		
	tps = torch.cat(tps, dim=0).unsqueeze(1)
	traj = torch.cat(traj, dim=2)
	Marker = np.hstack(Marker)#[:,:]
	
	latent_dim = traj.shape[3]

	# Fit pca
	latPCA = [None]*num_PCA
	for tr in range(num_PCA):
		latPCA[tr] = PCA(n_components=PCA_dim).fit(traj[:,tr].squeeze().reshape(-1, latent_dim).cpu())#.transform(traj.squeeze()[tr].cpu())
>>>>>>> fab8395ce21cb1139d04c7b66348fa4e9db98fe3

	# Apply PCA
	PCA_traj = [None]*num_PCA
	for tr in range(num_PCA):
<<<<<<< HEAD
		PCA_traj[tr] = latPCA.transform(traj.squeeze()[tr].cpu())

	return PCA_traj



def get_pca_fig(Trajectories):
	
=======
		PCA_traj[tr] = latPCA[tr].transform(traj.squeeze()[tr].cpu())[:,:,np.newaxis]

	PCA_traj = np.transpose( np.concatenate(PCA_traj, 2) , (2,0,1))

	return PCA_traj, tps.numpy(), Marker


def get_pca_fig(Trajectories):
	"""Not used anymore..."""

>>>>>>> fab8395ce21cb1139d04c7b66348fa4e9db98fe3
	samples, dim = Trajectories[0].shape

	fig = plt.figure()
	ax = fig.add_subplot(1, 1, 1)

	for tr in Trajectories:
<<<<<<< HEAD
		
=======

>>>>>>> fab8395ce21cb1139d04c7b66348fa4e9db98fe3
		if dim==1:
			ax.plot(tr[:,0], np.arange(samples), c='r', marker='1')

		elif dim==2:
			ax.plot(tr[:,0], tr[:,1], c='r', marker='1')

		elif dim==3:
			ax.plot(tr[:,0], tr[1], tr[:,2], c='r', marker='1')

<<<<<<< HEAD
	return fig

=======
	return fig
>>>>>>> fab8395ce21cb1139d04c7b66348fa4e9db98fe3
