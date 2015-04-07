from __future__ import division
import numpy as np
from ChalearnLAPSample import ActionSample
import os
import cv2
from helper_funcs import *
from sklearn.decomposition import RandomizedPCA
from scipy.ndimage.measurements import label
from scipy import ndimage
from sklearn.cluster import KMeans, SpectralClustering
from skimage import feature
import time
#from skimage.transform import PiecewiseAffineTransform, warp
from skimage import transform as tf
import math


def background_model(x_train, method='mean', n_components=10):
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
	# print sum(pca.explained_variance_ratio_)
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


def foreground_mask(back_vec, x, back_thres=.25):
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


def eigenback(back_vec, x, back_thres=.25, fore_thres=.1, rval='fore_mat', blur=False):
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
		# keep only regions containing at least <fore_thres> largest foreground region
		mask_size = sizes < (sizes.max() * fore_thres)
		remove_pixel = mask_size[label_im]
		img[remove_pixel] = 0
		if blur == True:
			img = cv2.GaussianBlur(img, (15, 15), 0)
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


def rotate_images(img_array, rotation):
	"""
	Assuming an image array is given
	:param img_array: a 3D array of images
	:param rotation: counter-clockwise rotation in radians (e.g. math.pi / 4)
	:return:
	"""
	# define a rotation matrix (used repeatedly later)
	rot_mat = np.array([
		[math.cos(rotation), math.sin(rotation) * -1],
		[math.sin(rotation), math.cos(rotation)]
	])
	# loop through each image and rotate it
	num_frames = img_array.shape[2]
	rotated_img_array = np.zeros(img_array.shape)
	for i in range(num_frames):
		img = img_array[:, :, i]
		# find how far the rotation will off-center the image:
		old_center = image_center(img)
		new_center = rot_mat.dot(old_center)
		shift = np.round(np.subtract(old_center, new_center))
		# transform the image
		tform = tf.SimilarityTransform(scale=1, rotation=rotation, translation=(shift[0], shift[1]))
		rotated_img_array[:, :, i] = tf.warp(img, tform)
		# another option that doesn't require manual translation:
		#from scipy.ndimage.interpolation import rotate
		#rotate(img, angle=30, reshape=True)
	return rotated_img_array


def randomly_scale_images(img_array):
	# loop through each image and rotate it
	num_frames = img_array.shape[2]
	scaled_img_array = np.zeros(img_array.shape)
	for i in range(num_frames):
		img = img_array[:, :, i]

		# find max amount that the image can be scaled without cropping
		# sum across rows to get column totals
		y_total = img.sum(axis=0)
		y_active = np.where(y_total > 0)[0]
		y_len = y_active.max() - y_active.min()
		# sum across columns to get row totals
		x_total = img.sum(axis=1)
		x_active = np.where(x_total > 0)[0]
		x_len = x_active.max() - x_active.min()
		scale_min = round(1. / (math.floor(min(img.shape[0]/x_len, img.shape[1]/y_len) * 10) / 10), 2)
		scale_range = np.hstack((np.arange(scale_min, .96, .01), np.arange(1.05, 2 - scale_min + .01, .01)))
		scale = random.choice(scale_range)

		# scale and save image
		scaled = scale_image(img, scale)
		scaled_img_array[:, :, i] = scaled
	return scaled_img_array


def scale_image(img, scale):
	# convert image to a mask in order to find center
	img_mask = img.copy()
	img_mask[img_mask > 0] = 1
	old_center = image_center(img_mask)
	new_center = np.round(old_center / scale)
	shift = np.round(np.subtract(new_center, old_center)).flatten()
	tform = tf.SimilarityTransform(scale=scale, translation=(shift[0], shift[1]))
	return tf.warp(img, tform)


def image_center(img):
	"""
	find the x,y coordinates of the weighted center of an image
	this is helpful for re-centering images that have been rotated
	or stretched
	:param img:
	:return:
	"""
	if not np.allclose(np.sort(np.unique(img)), np.array([0., 1.])):
		# convert image to a binary mask
		img[img > 0] = 1
	# what percent of pixels below to each row?
	y_weight = img.sum(axis=1) / img.sum()
	# multiply weight elementwise by the position of the pixel
	y_weighted_position = np.multiply(y_weight, range(len(y_weight)))
	# at what y-position are half of the pixels observed?
	y_center_of_mass = np.where(y_weighted_position.cumsum() > y_weighted_position.sum()/2)[0][0]
	# repeat for x-value
	x_weight = img.sum(axis=0) / img.sum()
	x_weighted_position = np.multiply(x_weight, range(len(x_weight)))
	x_center_of_mass = np.where(x_weighted_position.cumsum() > x_weighted_position.sum()/2)[0][0]
	return np.array([[x_center_of_mass], [y_center_of_mass]])


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
			n_clusters=30,
			n_components=50,
			back_thres=.25,
			fore_thres=.1,
			n_jobs=1):
		# must have parameters:
		self.back_vec = back_vec

		# optional parameters
		self.n_clusters = n_clusters
		self.n_components = n_components
		self.back_thres = back_thres
		self.fore_thres = fore_thres
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
		mask_mat = eigenback(self.back_vec, x, self.back_thres, self.fore_thres, rval='mask_mat', blur=True)
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
		print "Total explained variance:", sum(self.pca.explained_variance_ratio_)
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
		mask_mat = eigenback(self.back_vec, x, self.back_thres, self.fore_thres, rval='mask_mat', blur=True)
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







