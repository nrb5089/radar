import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from copy import deepcopy as dcp

M = 16
d = .1
ms = np.arange(M)
ry = ms*d - d*(M-1)/2
rx = -(d*ms - d*(M-1)/2)*np.sqrt(2)/2
rz = (d*ms - d*(M-1)/2)*np.sqrt(2)/2

def build_rvec(M,d):

	ms = np.arange(M)
	ry = ms*d - d*(M-1)/2
	rx = -(d*ms - d*(M-1)/2)*np.sqrt(2)/2
	rz = (d*ms - d*(M-1)/2)*np.sqrt(2)/2
	rvec = []
	for x,z in zip(rx,rz):
		current_set = []
		current_set.append(np.repeat(x,M))
		current_set.append(np.repeat(z,M))
		current_set.append(ry)
		#current_set1 = np.vstack(current_set).T
		rvec.append(np.vstack(current_set).T)
	rvec = np.vstack(rvec)
	#need to switch columns 1 and 2 so you get xyz
	rvec_hold = dcp(rvec)
	rvec[:,1] = rvec_hold[:,2]
	rvec[:,2] = rvec_hold[:,1]
	
	return rvec
if __name__ == '__main__':
	plt.close('all')
	fig,ax = plt.subplots(subplot_kw={'projection': '3d'})
# 	for x,z in zip(rx,rz):
# 		for y in ry:
# 			ax.plot(x,y,z,'kx',markersize = 10)
	rvec = build_rvec(16,.1)
	for r in rvec:
		ax.plot(r[0],r[1],r[2],'kx',markersize = 10)
	ax.set_ylabel('y')
	ax.set_xlabel('x')
	ax.set_zlabel('z')

	
	plt.show()