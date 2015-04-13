from __future__ import division
import numpy as np
import cv2
from helper_funcs import matrix_to_image, image_to_matrix
from sklearn.decomposition import RandomizedPCA
from scipy.ndimage.measurements import label
from scipy import ndimage


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

	# define background as an aggregation of each pixel value in the principal component space
	# can't see much of a difference between mean and median
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
	this uses the foreground_mask function above and a background vector
	created by the background_model function
	:param back_vec: a vector of the background image (use background_model to find this)
	:param x: a matrix with 1 row per image frame, each column represents a pixel.
		This algorithm is applied to this data and background is subtracted from each image
	:param back_thres: thresholding parameter for deciding what is a significant difference
		between the input image and the background.
		I.E what percent of the max background-foreground differences should be kept?
		.25 is a good default)
	:param fore_thres: thresholding parameter for deciding how big a region
		in the foreground needs to be in order to be accepted.
		The size of foreground regions is calculated based on how many pixels are touching
		divided the total number of pixels.
		fore_thres is the ratio of how big a region needs to be, relative to the
		largest region in order to be accepted.
		fore_thres = .1 => keep the largest region and any regions that are at least
		10% as a big as the largest region.
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
			img /= img.max()
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






