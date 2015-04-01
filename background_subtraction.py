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


def eigenback(back_vec, x, back_thres=.15, fore_thres=.25):
	"""
	This is a modified version of the eigenbackground algorithm
	this implements the functions that are defined above
	:param back_vec:
	:param x:
	:param back_thres:
	:param method:
	:return: the foreground matrix and mask
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
	fore_mask = image_to_matrix(mask_array)

	# apply foreground mask to each image in the entire video sequence
	fore_mat = np.multiply(fore_mask, x)
	return fore_mat


class Cluster(object):
	"""
	TBD
	Only doing Kmeans clustering right now
	"""
	def __init__(self, back_vec,
			n_clusters=20,
			n_components=25,
			back_thres=.15,
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
		fore_mat = eigenback(self.back_vec, x, self.back_thres, self.fore_thres)

		# map foreground to lower dimension for faster clustering
		self.pca = RandomizedPCA(n_components=self.n_components).fit(fore_mat)
		fore_pca = self.pca.transform(fore_mat)

		# apply unsupervised clustering on each image.
		self.cluster = KMeans(n_clusters=self.n_clusters, n_jobs=self.n_jobs).fit(fore_pca)
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
		fore_mat = eigenback(self.back_vec, x, self.back_thres, self.fore_thres)

		# map foreground to lower dimension
		fore_pca = self.pca.transform(fore_mat)

		# apply unsupervised clustering on each image.
		cluster_labels = self.cluster.predict(fore_pca)
		return cluster_labels







