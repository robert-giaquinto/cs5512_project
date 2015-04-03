from __future__ import division
import numpy as np
from ChalearnLAPSample import ActionSample
import os
import cv2
from helper_funcs import *
from sklearn.decomposition import RandomizedPCA
from scipy.ndimage.measurements import label
from scipy import ndimage
from sklearn import preprocessing
from sklearn.cluster import KMeans
from skimage import feature
import time


def background_model(x_train, method='mean', n_components=25):
	"""
	use data from x_train to create a model/image of the background
	:param x_train: a matrix with 1 row per image frame, each column represents a pixel
		PCA is trained on this data
	:return: a vector that represents the background image
	"""
	# clean the data before pca and clustering (subtract mean, divide by st. dev.)
	# scaler = preprocessing.StandardScaler().fit(x_train)
	# x_train = scaler.transform(x_train)
	# perform principal component analysis on the training images
	pca = RandomizedPCA(n_components=n_components).fit(x_train)
	train_pca = pca.transform(x_train)

	# define background as median pixel value in the principal component space
	if method == 'median':
		back_pca = np.median(train_pca, axis=0)
	elif method == 'mean':
		back_pca = np.mean(train_pca, axis=0)
	else:
		print "method must either be 'median' or 'mean'"
		return 1

	# transform to full sized matrix
	back_vec = pca.inverse_transform(back_pca)
	# add mean and variance back in
	# back_vec = scaler.inverse_transform(back_vec)
	return back_vec.reshape((1, len(back_vec)))


def foreground_mask(back_vec, x, back_thres=.15):
	"""
	Use a vector represent the background (see background_model function)
	to subtract the background from images.

	:param back_vec: a vector representing background model, each entry corresponds
		to a pixel in the background image
	:param x: a matrix with 1 row per image frame, each column represents a pixel
		method is applied to this data and background is subtracted
	:param back_thres: thresholding parameter. 1 => keep all pixels in image that are
		different than background model. 0.0001 => only keep pixel that's most
		different from background.
	:return: fore_mask a binary matrix, each row corresponds to the image from the same row
		in x_test. columns are yes or no saying whether the pixel is in foreground or not
	"""
	# subtract background from the test images
	fore_mask = np.abs(np.subtract(x, back_vec))

	# thresholding: keep only differences that are 'significant'
	thres = back_thres * fore_mask.max(axis=1).reshape((fore_mask.shape[0], 1))
	fore_mask[np.less(fore_mask, thres)] = 0
	# set every significant difference to True
	fore_mask[fore_mask > 0] = 1
	return fore_mask


def eigenback(back_vec, x, back_thres=.15, fore_thres=.25, rval='fore_mat'):
	"""
	This is a modified version of the eigenbackground algorithm
	this implements the functions that are defined above
	:param back_vec:
	:param x:
	:param back_thres: thresholding parameter. 1 => keep all pixels in image that are
		different than background model. 0.0001 => only keep pixel that's most
		different from background.
	:param fore_thres: thresholding parameter for deciding how big a region in the foreground
		needs to be in order to not be removed.
	:return: the foreground matrix or mask
	"""
	# use background image to create a foreground mask
	fore_mask = foreground_mask(back_vec, x, back_thres=back_thres)

	# only keep continuously connected parts of the mask
	mask_array = matrix_to_image(fore_mask)
	for i in range(fore_mask.shape[0]):
		img = mask_array[:, :, i]
		# assign a label to each pixel in a connected region (including diagonal connections)
		label_im, nb_labels = label(img, structure=np.array([[1,1,1], [1,1,1], [1,1,1]]))
		# how many pixels are in each region?
		sizes = ndimage.sum(img, label_im, range(nb_labels + 1))
		# keep only regions containing at least <fore_thres> of the total foreground
		mask_size = sizes < (img.sum() * fore_thres)
		remove_pixel = mask_size[label_im]
		img[remove_pixel] = 0
		mask_array[:, :, i] = img

	if rval == 'fore_mat':
		# apply foreground mask to each image in the entire video sequence
		fore_mask = image_to_matrix(mask_array)
		fore_mat = np.multiply(fore_mask, x)
		return fore_mat
	elif rval == 'mask_mat':
		# just return the mask
		fore_mask = image_to_matrix(mask_array)
		return fore_mask
	elif rval == 'mask_array':
		# return mask as an array
		return mask_array
	else:
		print 'Must specify rval to be either fore_mat, mask_mat, or mask_array. returning foreground mask by default'
		fore_mask = image_to_matrix(mask_array)
		return fore_mask


def detect_edges(mask_array, sigma=3):
	"""

	:param mask_array: a 3D image array of foreground masks
	:return:
	"""
	num_images = mask_array.shape[2]
	edge_array = np.zeros(mask_array.shape)
	for i in range(num_images):
		edge_array[:,:, i] = feature.canny(mask_array[:,:, i], sigma=sigma)
	return edge_array


class Cluster(object):
	"""
	TBD
	Only doing Kmeans clustering right now
	Before clustering each image has:
		1. background subtracted
		2. only contiguous regions of foreground are kept
		skip: ---# 3. edges of foreground are extracted
		4. foreground is mapped to a lower dimension using PCA
		5. finally, the images are clustered
	"""
	def __init__(self, back_vec,
			n_clusters=20,
			n_components=30,
			back_thres=.20,
			fore_thres=.25,
			normalize=False,
			n_jobs=1):
		# must have parameters:
		self.back_vec = back_vec

		# optional parameters
		self.n_clusters = n_clusters
		self.n_components = n_components
		self.back_thres = back_thres
		self.fore_thres = fore_thres
		self.normalize = normalize
		self.n_jobs = n_jobs

		# to be assigned in as later
		self.cluster = None
		self.pca = None

	def fit(self, x):
		"""
		:param x: an image matrix to train a clustering algorithm on
		"""
		# use the background image given to extract foreground from x
		tstart = time.time()
		mask_mat = eigenback(self.back_vec, x, self.back_thres, self.fore_thres, rval='mask_mat')
		# tend = time.time()
		# print "foreground mask extracted in", tend - tstart, "seconds."
		# TOO SLOW, TRY SOMETHING SIMPLER:
		# mask_mat = foreground_mask(self.back_vec, x, self.back_thres)
		tend = time.time()
		print "foreground mask extracted in", tend - tstart, "seconds."

		# detect edges on foreground, convert to a matrix
		# tstart = time.time()
		# fore_edges = image_to_matrix(detect_edges(mask_array, sigma=3))
		# tend = time.time()
		# print "edges extracted in", tend - tstart, "seconds."

		# map foreground to lower dimension for faster clustering
		tstart = time.time()
		self.pca = RandomizedPCA(n_components=self.n_components).fit(mask_mat)
		fore_pca = self.pca.transform(mask_mat)
		tend = time.time()
		print "edges mapped to a lower dimension in", tend - tstart, "seconds."

		# apply unsupervised clustering on each image.
		tstart = time.time()
		self.cluster = KMeans(n_clusters=self.n_clusters, n_jobs=self.n_jobs).fit(fore_pca)
		tend = time.time()
		print "edges clustered in", tend - tstart, "seconds."
		return self

	def predict(self, x):
		"""
		SHOULD RETURN IMAGE FOREGROUND TOO?
		:param x: lookup nearest cluster based on images used to train model (from fit)
		:return: clustering labels, one for each row of x
		"""
		if self.cluster is None or self.pca is None:
			print "fit method must be called first"
			return 1

		# use the background image given to extract foreground from x
		tstart = time.time()
		mask_mat = eigenback(self.back_vec, x, self.back_thres, self.fore_thres, rval='mask_mat')
		# tend = time.time()
		# print "foreground mask extracted in", tend - tstart, "seconds."
		# TOO SLOW, TRY SOMETHING SIMPLER:
		# mask_mat = foreground_mask(self.back_vec, x, self.back_thres)
		tend = time.time()
		print "foreground mask extracted in", tend - tstart, "seconds."

		# detect edges on foreground, convert to a matrix
		# tstart = time.time()
		# fore_edges = image_to_matrix(detect_edges(mask_array, sigma=3))
		# tend = time.time()
		# print "edges extracted in", tend - tstart, "seconds."

		# map foreground to lower dimension using trained pca
		tstart = time.time()
		fore_pca = self.pca.transform(mask_mat)
		tend = time.time()
		print "edges mapped to a lower dimension in", tend - tstart, "seconds."

		# apply unsupervised clustering on each image using trained model
		tstart = time.time()
		cluster_labels = self.cluster.predict(fore_pca)
		tend = time.time()
		print "edges clustered in", tend - tstart, "seconds."
		return cluster_labels







