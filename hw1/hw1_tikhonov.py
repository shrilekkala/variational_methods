#!/Users/conda/anaconda3/envs/sp/bin/python
# Replace above with path to python

import skimage, skimage.io
import scipy, scipy.stats
import numpy
import math
import matplotlib.pyplot as plt


# hw1_tikhonov.py - Template Python source code for Problem (1) on HW1 in CAAM/Stat 31240
# (Variational Methods in Image Processing)
#
# Description: In this problem, the goal is to implement a variant of a Tikhonov-Arsenin
# regularization scheme based on the L^2 norm of an approximate gradient of the image.

def read_image(filename):
    return skimage.io.imread(filename)


def write_image(im, filename):
    skimage.io.imsave(filename, im)


def display_image(im, mode='bw'):
    if mode == 'bw':
        plt.imshow(im, cmap=plt.cm.gray)
    elif mode == 'color':
        plt.imshow(im)


# vec(X) returns a column vector with the columns of
# the matrix X concatenated sequentially
def vec(X):
    dim = X.shape
    length = X.size
    return X.transpose().reshape(X.size, 1)


# mat(X,shape) inverts the operation vec(X), i.e. it returns a matrix such that mat(vec(X),X.shape)=X
def mat(v, shape):
    return v.reshape(shape).transpose()


# compute_Q(M) routine computes the matrix which corresponds to a finite
# difference scheme for approximating the gradient of an image composed 
# of M\times M pixels, (1) arranged in a matrix u=(u_{ij}), and then (2) 
# converted to a vector representation with u_vec=vec(u), using 
# the vec() routine above.
#
# Once Q is computed, the quantity |Q*u_vec|^2 is an approximation 
# to \lVert\nabla u\rVert_{L^2}^2.
#
# For discussion of the difference quotient derivation, see, e.g. the excerpt from the book
# of Hansen, Nagy, and O'Leary linked in the Canvas assignment.  The relevant
# sections are 4.4 and 7.3.
def compute_Q(dim):
    diagonals = numpy.zeros([2, dim])
    diagonals[0, :] = -1
    diagonals[1, :] = 1
    diagonals[0, dim - 1] = 0
    D_x = scipy.sparse.kron(scipy.sparse.eye(dim), scipy.sparse.spdiags(diagonals, [0, 1], dim, dim))
    D_y = scipy.sparse.kron(scipy.sparse.spdiags(diagonals, [0, 1], dim, dim), scipy.sparse.eye(dim))
    return scipy.sparse.vstack([D_x, D_y])


def compute_tikhonov_arsenin(im, lmbda):
    dim = im.shape
    im_copy = vec(im.copy())
    Q = compute_Q(im.shape[0])

    # FILL IN: Compute the Tikhonov-Arsenin regularization using the
    # discretized gradient values stored in the multi-dimensional
    # array Q.  To use the Q computed above, you may have to look up
    # and use the scipy.sparse routines (see also scipy.sparse.linalg.spsolve).

    # FILL IN:
    # identity matrix
    I = scipy.sparse.eye(dim[0] * dim[1])
    # form matrices A and b to solve for Av  = b
    A = I + lmbda * Q.T @ Q
    b = im_copy
    solution = scipy.sparse.linalg.spsolve(A, b)

    # The below line converts a column vector ``solution'' back to the matrix image format
    # we've been using.
    im_smoothed = mat(solution, dim)

    return im_smoothed


def main():
    # FILL IN: Load the clock image file into the variable im_input.
    im_input = read_image('5.1.12.tiff')
    ###

    # Set im to the image in {0,...,255} pixel value format.
    im = skimage.util.img_as_ubyte(im_input)

    plt.figure()
    plt.subplot(1, 3, 1)
    plt.title('Original Clock')
    display_image(im, 'bw')

    # FILL IN: Create a noisy copy of the image im and store
    # this in the variable im_noisy.

    # Using skimage.util.random_noise function to add Gaussian noise to the image
    im_noisy = skimage.util.random_noise(im,
                                         mode='gaussian',
                                         mean=0,
                                         var=0.05)
    ###

    plt.subplot(1, 3, 2)
    plt.title('Noisy Clock')
    display_image(im_noisy, 'bw')

    # lambda is set to 20
    im_smoothed = compute_tikhonov_arsenin(im_noisy, 20)

    plt.subplot(1, 3, 3)
    plt.title('Tikhonov-Arsenin')
    display_image(im_smoothed, 'bw')

    plt.show()


if __name__ == "__main__":
    main()
