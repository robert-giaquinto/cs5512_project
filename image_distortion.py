from __future__ import division
import math
import random
import numpy as np
import scipy.ndimage.interpolation as im
from helper_funcs import image_to_matrix


def distorted_image_set(imgs, num_runs=10, data_dir=None, filename='distorted_matrix.csv', verbose=True):
	"""
	Take a set of images and randomly distort them num_runs times.
	The result should be a training data set that is more robust to new images

	:param imgs: a 3D array of images to be distorted and turned into a training data set
	:param labels: a vector of classification labels for each image in imgs
	:param num_runs: how many copies of the data set should be made
	:return: x, y
		x = the training data set which has each image in imgs distorted num_runs
			times, and transformed into a matrix
		y = the labels for each image, repeated num_runs times
	"""
	# convert x_train to an image array
	num_imgs = imgs.shape[2]
	if data_dir is None:
		distorted_matrix = np.zeros([num_imgs * (num_runs+1), imgs.shape[0] * imgs.shape[1]])
		# append original images too
		distorted_matrix[0:num_imgs, :] = image_to_matrix(imgs)

	for r in range(num_runs):
		distorted = image_to_matrix(randomly_distort_images(imgs))
		if data_dir is not None:
			if r == 0:
				# also save original
				np.savetxt(data_dir + filename, image_to_matrix(imgs), fmt='%1.3f', delimiter=",")
			# save result to disk (add to current results
			with open(data_dir + filename, 'a') as f_handle:
				np.savetxt(f_handle, distorted, fmt='%1.3f', delimiter=",")
		else:
			# store in a matrix
			start_index = (r+1) * num_imgs
			end_index = start_index + num_imgs
			distorted_matrix[start_index:end_index, :] = distorted
		if verbose:
			print "done with run", r

	if data_dir is None:
		return distorted_matrix


def randomly_distort_images(img_array):
	"""
	1. This function takes an image array and, for each image, the image is rotated
	by some random amount between 45 and -45 degrees,
	each image is then returned as part of an imaged array.

	2.  function takes an image array and, for each image, finds the most that it
	could be enlarged without cropping the image. The image is then re-scaled
	by some random amount and returned as part of an imaged array

	3. function translates image to a random position without cropping

	:param img_array: a 3D array of images
	:return: a 3D image array where each image from input has been rotated by
		some reasonable amount.
	"""
	# loop through each image and rotate it
	num_frames = img_array.shape[2]
	distorted_img_array = np.zeros(img_array.shape)
	for i in range(num_frames):
		# 0. begin by centering the foreground in the middle of the frame
		img = img_array[:, :, i]
		old_center = image_center(img)  # (row, col)
		frame_center = np.array([[img.shape[0]/2], [img.shape[1]/2]])
		translation = np.round(np.subtract(frame_center, old_center)).flatten()
		img = im.shift(img, (translation[0], translation[1]))

		# 1. randomly select a rotation amount
		# randomly select a rotation angle between -40 to 41 degrees
		rotation_range = (np.arange(-30, 31, 1)/360) * 2 * math.pi
		rotation = random.choice(rotation_range)
		# rotate the image and save the result
		# and normalize result to be zeros and ones
		rotated_img = normalize(im.rotate(img, (rotation * 360 / (math.pi * 2)), reshape=False))

		# 2. randomly select a reasonable scaling amount
		# find max amount that the image can be scaled without cropping
		# sum across columns to find row active in foreground
		row_total = rotated_img.sum(axis=1)
		row_active = np.where(row_total > 0)[0]
		# what is range of active pixels?
		row_len = row_active.max() - row_active.min()
		# sum across rows to find columns active in foreground
		col_total = rotated_img.sum(axis=0)
		col_active = np.where(col_total > 0)[0]
		# what is range of active pixels?
		col_len = col_active.max() - col_active.min()
		# at very most, increase by half (and never get within 10% of border)
		scale_max = min(1.5, math.floor(min(img.shape[0]/(1.1*row_len), img.shape[1]/(1.1*col_len)) * 10) / 10)
		# at most, shrink by a half
		scale_range = np.arange(.5, scale_max + .01, .01)
		if len(scale_range) != 0:
			scale = round(random.choice(scale_range), 2)
			# apply transformation and normalize results
			rotated_scaled_img = normalize(scale_image(rotated_img, scale))
		else:
			rotated_scaled_img = rotated_img

		# 3. randomly translate the image such that no cropping occurs
		# y_shift = reasonable translations along vertical axis
		row_total = rotated_scaled_img.sum(axis=1)
		row_active = np.where(row_total > 0)[0]
		row_shift_range = np.hstack((
			np.arange(-1, -1 * row_active.min(), -5),
			np.arange(1, img.shape[0] - row_active.max(), 5)
		))
		# x_shift = reasonable translations along horizontal axis
		col_total = rotated_scaled_img.sum(axis=0)
		col_active = np.where(col_total > 0)[0]
		col_shift_range = np.hstack((
			np.arange(-1, -1 * col_active.min(), -5),
			np.arange(1, img.shape[1] - col_active.max(), 5)
		))
		# randomly select a pair of reasonable shifts in either direction
		if len(row_shift_range) != 0:
			row_shift = random.choice(row_shift_range)
		else:
			row_shift = 0
		if len(col_shift_range) != 0:
			col_shift = random.choice(col_shift_range)
		else:
			col_shift = 0
		distorted_img = normalize(im.shift(rotated_scaled_img, (row_shift, col_shift)))

		# save final results
		distorted_img_array[:, :, i] = distorted_img
	return distorted_img_array


def scale_image(img, scale):
	"""
	Low level function to rescale a single image.
	This function also finds the necessary translation such that the image
	is still centered after being re-scaled
	:param img: an image of size x-pixels by y-pixels
	:param scale:  scaling factor
	:return:
	"""
	new_img = np.round(im.zoom(img, zoom=scale), decimals=4)
	if scale < 1:
		# pad
		x_start = math.floor((img.shape[0] - new_img.shape[0])/2)
		y_start = math.floor((img.shape[1] - new_img.shape[1])/2)
		rval = np.zeros(img.shape)
		rval[x_start:(x_start + new_img.shape[0]), y_start:(y_start + new_img.shape[1])] = new_img
	else:
		# crop
		x_start = (new_img.shape[0] - img.shape[0])/2
		y_start = (new_img.shape[1] - img.shape[1])/2
		rval = new_img[x_start:(x_start + img.shape[0]), y_start:(y_start + img.shape[1])]
	return rval


def image_center(img):
	"""
	find the x,y coordinates of the weighted center of an image
	this is helpful for re-centering images that have been rotated
	or stretched
	:param img: a foreground image, dimensions = x-pixels by y-pixels
	:param weighted: if true then it gives center of mass, otherwise it's unweighted
	:return:
	"""
	# what percent of pixels below to each row?
	foreground_size = np.count_nonzero(img)
	row_weight = (img != 0).sum(axis=1) / foreground_size
	# at what y-position are half of the pixels observed?
	row_center_of_mass = np.where(row_weight.cumsum() >= 0.5)[0][0]
	# repeat for x-value
	col_weight = (img != 0).sum(axis=0) / foreground_size
	col_center_of_mass = np.where(col_weight.cumsum() >= 0.5)[0][0]
	return np.array([[row_center_of_mass], [col_center_of_mass]])


def normalize(img):
	# normalize the result
	img = img - img.min()
	img = img / img.max()
	img[img > .5] = 1
	img[img < 1] = 0
	return img