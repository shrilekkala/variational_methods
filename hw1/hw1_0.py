#!/Users/conda/anaconda3/envs/sp/bin/python
#Replace above with path to python

import skimage, skimage.io
import scipy, scipy.stats
import numpy
import math
import matplotlib.pyplot as plt

# hw1_0.py - Template Python source code for Problem (0) on HW1 in CAAM/Stat 31240
# (Variational Methods in Image Processing)
#
# This example Python program does the following:
#
# (1) Read in the Clock (B+W) and San Diego (Color) images from files.
# (2) Load the Photographer (B+W, pixel values in the range 0..255) image from the 
#     scikit.data package.
# (3) Convert the San Diego image to greyscale (floating point pixel values), and 
#     further convert that to integer pixel values in the range 0..255 (data 
#     type: uint8 -- 8 bit unsigned integers).
# (4) Display each of these images (Clock, 3 versions of San Diego, Photographer)
#     in separate figure windows.  The management of figures is very similar to
#     the mechanism used in Matlab.
# (5) Add random Gaussian noise to the San Diego 0..255 image, and display the 
#     original/noisy results.  The function add_noise defined here includes three 
#     strategies to help us see how to start to manipulate images, as well as how 
#     to apply some of the pre-written algorithms included in the scikit-image 
#     package.  
#
#     The strategy to use is selected by a function parameter.  This is one way
#     you can use the source code you write as part of a broader strategy to
#     organize and communicate your ideas.

def read_image(filename):
	return skimage.io.imread(filename)

def write_image(im, filename):
	skimage.io.imsave(filename,im)

def display_image(im,mode='bw'):
	if mode=='bw':
		plt.imshow(im,cmap=plt.cm.gray)
	elif mode=='color':
		plt.imshow(im)

def add_noise(im,variance,mode='by_hand_vectorized'):
	if mode=='by_hand_for_loop':
		copy=skimage.util.img_as_ubyte(im.copy())
		dim=copy.shape 
		noise=scipy.stats.norm.rvs(size=copy.size,
			scale=math.sqrt(variance)*255)
		for i in range(0,dim[0]):
			for j in range(0,dim[1]):
				copy[i,j]+=int(next(noise))
				copy[i,j]=max(copy[i,j],0)
				copy[i,j]=min(copy[i,j],255)
		return copy
	elif mode=='by_hand_vectorized':
		copy=skimage.util.img_as_ubyte(im.copy())
		noise=scipy.stats.norm.rvs(size=copy.size,
			scale=math.sqrt(variance)*255).astype('int16')
		copy=copy.astype('int16')+numpy.reshape(noise,copy.shape)
		copy=numpy.maximum(copy,numpy.zeros(copy.shape))
		copy=numpy.minimum(copy,255*numpy.ones(copy.shape))
		copy=copy.astype('uint8')
		return copy
	elif mode=='built_in':
		noisy=skimage.util.random_noise(im,mode='gaussian',
			mean=0,var=variance)
		return skimage.util.img_as_ubyte(noisy)

def main():
	im_sd_color=read_image('2.2.23.tiff') # San Diego; color
	im_clock=read_image('5.1.12.tiff') # Clock; b&w

	fetch_function=getattr(skimage.data,'camera')
	im_photographer=fetch_function()

	im_sd_bw=skimage.color.rgb2gray(im_sd_color)
	write_image(im_sd_bw,'sd_bw.tif')

	im_sd_bw_255=skimage.util.img_as_ubyte(im_sd_bw,force_copy=True)
	write_image(im_sd_bw_255,'sd_bw.gif')

	plt.figure()
	plt.title('Clock')
	display_image(im_clock)

	plt.figure()
	plt.title('San Diego')
	plt.subplot(1,3,1)
	display_image(im_sd_color,'color')
	plt.subplot(1,3,2)
	display_image(im_sd_bw,'bw')
	plt.subplot(1,3,3)
	display_image(im_sd_bw_255,'bw')

	plt.figure()
	plt.title('Photographer')
	display_image(im_photographer)

	noisy=add_noise(im_sd_bw_255,0.05,'built_in')
	plt.figure()
	plt.subplot(1,2,1)
	display_image(im_sd_bw_255,'bw')
	plt.subplot(1,2,2)
	display_image(noisy,'bw')

	plt.show()

if __name__=="__main__":
	main()
