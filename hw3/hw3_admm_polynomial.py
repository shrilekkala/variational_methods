#!/Users/conda/anaconda3/envs/sp/bin/python
#Replace above with path to python

import skimage, skimage.io
import scipy, scipy.stats
import numpy
import math
import matplotlib.pyplot as plt

def update_x(z,y,rho,f,A,lmbda):
	## FILL IN

def update_z(x,y,rho,lmbda):
	## FILL IN

def update_y(x,y,z,rho,lmbda):
	## FILL IN

def compute_admm_lasso(f,A,lmbda,rho):
	dim=A.shape
	m=dim[0]
	n=dim[1]

	x=numpy.zeros([n])
	z=numpy.zeros([n])
	y=numpy.zeros([n])

	count=1
	while True:
		z_last=z.copy()

		x=update_x(z,y,rho,f,A,lmbda)
		z=update_z(x,y,rho,lmbda)
		y=update_y(x,y,z,rho,lmbda)

		print(count,": stopping_criteria: ",numpy.linalg.norm(x-z)," ",rho*numpy.linalg.norm(z-z_last))

		if ( numpy.linalg.norm(x-z)<8e-5 ) and ( rho*numpy.linalg.norm(z-z_last)<1e-6 ):
			break
		count=count+1
	
	print("ADMM iteration complete: ",count,"iterations.")

	return x

def main():
	# Generate ``noisy'' data.
	x0=numpy.linspace(0,10,10)
	f=2/25*x0*(x0-5)*(x0-10)+numpy.random.uniform(-0.375,0.375,x0.shape)

	# Assemble matrix for polynomial regression.
	degree=10
	x1=numpy.linspace(0,10,100)
	A=numpy.ndarray([x0.size,degree])
	B=numpy.ndarray([x1.size,degree])
	A[:,0]=numpy.ones([1,x0.size])
	B[:,0]=numpy.ones([1,x1.size])
	for i in range(1,degree):
		A[:,i]=numpy.power(x0,i)
		B[:,i]=numpy.power(x1,i)

	poly_regression=numpy.linalg.solve(numpy.matmul(A.transpose(),A),numpy.matmul(A.transpose(),f))
	poly_y=numpy.matmul(B,poly_regression)

	plt.subplot(1,2,1)
	plt.title('Polynomial Regression')
	plt.ylim(-7,7)
	plt.plot(x0,f,'.')
	plt.plot(x1,poly_y)

	lasso_poly_regression=compute_admm_lasso(f,A,0.12,.10)
	lasso_y=numpy.matmul(B,lasso_poly_regression)

	plt.subplot(1,2,2)
	plt.title('Lasso Regularized Regression via ADMM')
	plt.ylim(-7,7)
	plt.plot(x0,f,'.')
	plt.plot(x1,lasso_y)
		
	plt.show()

if __name__=="__main__":
	main()
