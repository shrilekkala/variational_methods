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

def compute_gradient(u):
	dim=u.shape
	p=numpy.zeros([dim[0],dim[1],2])

	p[0:(dim[0]-1),:,0]=u[1:dim[0],:]-u[0:(dim[0]-1),:]
	p[dim[0]-1,:,0]=numpy.zeros([dim[1]])

	p[:,0:(dim[1]-1),1]=u[:,1:dim[1]]-u[:,0:(dim[1]-1)]
	p[:,dim[1]-1,1]=numpy.zeros([dim[0]])

	return p

def compute_div(p):
	dim=p.shape
	div=p.copy()

	#div[0,:,0] stays the same
	div[1:(dim[0]-1),:,0]=div[1:(dim[0]-1),:,0]-p[0:(dim[0]-2),:,0]
	div[dim[0]-1,:,0]=-p[dim[0]-2,:,0]

	#div[:,0,1] stays the same
	div[:,1:(dim[1]-1),1]=div[:,1:(dim[1]-1),1]-p[:,0:(dim[1]-2),1]
	div[:,dim[1]-1,1]=-p[:,dim[1]-2,1]

	return div[:,:,0]+div[:,:,1]

def compute_tv_chambolle(im,lmbda,tau=1/8,p0=None,tol=1/100):
	dim=im.shape
	f=im.copy()

	if(p0==None):
		p0=numpy.zeros([dim[0],dim[1],2])

	p=p0.copy()
	count=1
	while True:
		p_last=p

		divp=compute_div(p_last)
		step=compute_gradient(divp-(1/lmbda)*f)

		## FILL IN BELOW (you may use more than one line, of course)
		p = (p_last + tau * step) / (1 + tau * numpy.abs(step))
	
		stopping_criteria=numpy.amax(numpy.abs((p-p_last).reshape(p.size,1)))
		print(count,": stopping_criteria=",stopping_criteria)

		if ( stopping_criteria < tol ):
			break
		count=count+1
	
	print("Chambolle iteration complete: ",count,"iterations.")
	im_smoothed=f-lmbda*compute_div(p)

	return im_smoothed

def main():
	im_input=read_image('5.1.12.tiff')
	im_input=read_image('5.2.10.tiff')


	im=skimage.util.img_as_ubyte(im_input)

	plt.figure()
	plt.subplot(1,3,1)
	plt.title('Original Image')
	display_image(im,'bw')

	im_noisy=skimage.util.img_as_ubyte(im.copy())
	noise=scipy.stats.norm.rvs(size=im_noisy.size,
		scale=math.sqrt(0.005)*255).astype('int16')
	im_noisy=im_noisy.astype('int16')+numpy.reshape(noise,im_noisy.shape)
	im_noisy=numpy.maximum(im_noisy,numpy.zeros(im_noisy.shape))
	im_noisy=numpy.minimum(im_noisy,255*numpy.ones(im_noisy.shape))
	im_noisy=im_noisy.astype('uint8')

	plt.subplot(1,3,2)
	plt.title('Noisy Image')
	display_image(im_noisy,'bw')

	im_smoothed=compute_tv_chambolle(im_noisy,15,tol=1/1000)
	
	plt.subplot(1,3,3)
	plt.title('ROF - Chambolle (dual problem)')
	display_image(im_smoothed,'bw')

	plt.show()

if __name__=="__main__":
	main()
