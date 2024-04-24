#!/Users/conda/anaconda3/envs/sp/bin/python
#Replace above with path to python

import skimage, skimage.io
import scipy, scipy.stats
import numpy
import math
import matplotlib.pyplot as plt

def read_image(filename):
	return skimage.io.imread(filename)

def write_image(im, filename):
	skimage.io.imsave(filename,im)

def display_image(im,mode='bw'):
	if mode=='bw':
		plt.imshow(im,cmap=plt.cm.gray)
	elif mode=='color':
		plt.imshow(im)

def vec(X):
	dim=X.shape
	length=X.size
	return X.transpose().reshape(X.size,)

def mat(v,shape):
	return v.reshape(shape).transpose()

def compute_Q(dim):
	diagonals=numpy.zeros([2,dim])
	diagonals[0,:]=-1
	diagonals[1,:]=1
	diagonals[0,dim-1]=0
	D_x=scipy.sparse.kron(scipy.sparse.eye(dim),scipy.sparse.spdiags(diagonals, [0,1], dim, dim))
	D_y=scipy.sparse.kron(scipy.sparse.spdiags(diagonals, [0,1], dim, dim),scipy.sparse.eye(dim))
	return scipy.sparse.vstack([D_x,D_y])

def update_x(z,y,rho,f,Q,lmbda):
	# Get matrix M and vector b in Mx = b
	M = scipy.sparse.eye(Q.shape[1]) + rho * Q.T @ Q
	b = f + Q.T @ (rho * z - y)
	x = scipy.sparse.linalg.spsolve(M,b)
	return x

def update_z(x,y,rho,Q,lmbda):
	condition_vec = y / rho + Q @ x

	# initialize z
	z = numpy.zeros(Q.shape[0])

	# Index of elements for soft thresholding
	upper = numpy.where(condition_vec > lmbda / rho)
	lower = numpy.where(condition_vec < - lmbda / rho)

	# adjust elements of z
	z[upper] = condition_vec[upper] - lmbda / rho
	z[lower] = condition_vec[lower] + lmbda / rho

	return z

def update_y(x,y,z,rho,Q,lmbda):
	y = y + rho * (Q @ x - z)
	return y

def compute_tv_admm(im,lmbda,rho):
	dim=im.shape
	im_copy=vec(im.copy())
	Q=compute_Q(im.shape[0])

	Q=Q.tocsc()

	x=im_copy.copy()
	z=Q@x
	y=0*z.copy()

	count=1
	while True:
		z_last=z.copy()
		x=update_x(z,y,rho,im_copy,Q,lmbda)
		z=update_z(x,y,rho,Q,lmbda)
		y=update_y(x,y,z,rho,Q,lmbda)

		print(count,": stopping criteria: ",numpy.linalg.norm(Q@x-z)," ",rho*numpy.linalg.norm(Q.transpose()@(z-z_last)))

		if ( numpy.linalg.norm(Q@x-z)<math.sqrt(z.size)*0.5) and ( rho*numpy.linalg.norm(Q.transpose()@(z-z_last))<math.sqrt(x.size)*0.5):
			break
		count=count+1
	
	print("ADMM iteration complete: ",count,"iterations.")

	im_smoothed=mat(x,dim)

	return im_smoothed

def main():
	im_input=read_image('5.1.12.tiff') 

	im=skimage.util.img_as_ubyte(im_input)

	plt.figure()
	plt.subplot(1,3,1)
	plt.title('Original Clock')
	display_image(im,'bw')

	im_noisy=skimage.util.img_as_ubyte(im.copy())
	noise=scipy.stats.norm.rvs(size=im_noisy.size,
		scale=math.sqrt(0.005)*255).astype('int16')
	im_noisy=im_noisy.astype('int16')+numpy.reshape(noise,im_noisy.shape)
	im_noisy=numpy.maximum(im_noisy,numpy.zeros(im_noisy.shape))
	im_noisy=numpy.minimum(im_noisy,255*numpy.ones(im_noisy.shape))
	im_noisy=im_noisy.astype('uint8')

	plt.subplot(1,3,2)
	plt.title('Noisy Clock')
	display_image(im_noisy,'bw')

	im_smoothed=compute_tv_admm(im_noisy,20,1)
	
	plt.subplot(1,3,3)
	plt.title('ROF Regularization (ADMM)')
	display_image(im_smoothed,'bw')

	plt.show()

if __name__=="__main__":
	main()
