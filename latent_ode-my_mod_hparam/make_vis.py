
#Author: Nando Metzger

import numpy as np
import pickle
import os

import matplotlib.pyplot as plt
from scipy import signal

from sklearn.linear_model import LinearRegression

def traj_plot(traj1, times1, traj2, times2, dim):

	fig = plt.figure()
	ax = fig.add_subplot(1, 1, 1)

	#First Model
	colors = ['mediumslateblue' ,'blue', 'mediumblue', 'midnightblue']
	num_traj = traj1.shape[0]
	num_traj = 1

	legend = []

	for t in range(num_traj):
		tr = traj1[t]

		# detrend signal
		tr = tr - LinearRegression().fit(times1, tr).predict(times1)

		if dim==1:
			#ax.plot(times1, tr[:,0] , c=colors[t%num_traj], marker='1', label="ODE-RNN: Sample "+str(t+1) )
			ax.plot(times1, tr[:,0] , c=colors[t%num_traj], marker='1', label="ODE-GRU")
			plt.ylabel("PCAaxis")
			plt.xlabel("Time")
			#legend.append("ODE-RNN 1")

		elif dim==2:
			ax.plot(tr[:,0], tr[:,1], c='r', marker='1')

		elif dim==3:
			ax.plot(tr[:,0], tr[1], tr[:,2], c='r', marker='1')


	#Second Model
	colors = ['tomato' ,'orangered', 'red', 'darkred']
	num_traj = traj2.shape[0]
	num_traj = 1

	for t in range(num_traj):
		tr = traj2[t]

		#detrend signal
		tr = tr - LinearRegression().fit(times2, tr).predict(times2)

		if dim==1:
			#ax.plot(times2, tr[:,0] , c=colors[t%num_traj], marker='1', label="RNN: Sample "+str(t+1))
			ax.plot(times2, tr[:,0] , c=colors[t%num_traj], marker='1', label="RNN")
			plt.ylabel("PCAaxis")
			plt.xlabel("Time")

		elif dim==2:
			ax.plot(tr[:,0], tr[:,1], c='r', marker='1')

		elif dim==3:
			ax.plot(tr[:,0], tr[1], tr[:,2], c='r', marker='1')

	# show the figure

	handles, labels = ax.get_legend_handles_labels()

	# reverse the order
	ax.legend()
	return fig


if __name__ == '__main__':

	ExperimentID1 = 9284000 #ODERNN
	ExperimentID2 = 3100000 #RNN

	root = "vis"
	with open( os.path.join(root, "traj_dict" + str(ExperimentID1) + ".pickle"),'rb' ) as file:
		traj_dict1 = pickle.load(file)
		print(1)

	with open( os.path.join(root, "traj_dict" + str(ExperimentID2) + ".pickle"), 'rb' ) as file:
		traj_dict2 = pickle.load(file)
		print(2)

	#1D
	traj1, times1 = traj_dict1["PCA_trajs1"]
	traj2, times2 = traj_dict2["PCA_trajs1"]
	
	#2D
	#traj1, times1 = traj_dict1["PCA_trajs2"]
	#traj2, times2 = traj_dict2["PCA_trajs2"]
	
	#3D
	#traj1, times1 = traj_dict1["PCA_trajs3"]
	#traj2, times2 = traj_dict2["PCA_trajs3"]
	
	traj_plot(traj1, times1, traj2, times2, dim=1)
	
	plt.show()
	#samples, dim = Trajectories[0].shape

