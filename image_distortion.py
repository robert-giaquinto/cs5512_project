from __future__ import division
import math
import random
import numpy as np
from skimage import transform as tf
from helper_funcs import matrix_to_image, image_to_matrix
from background_subtraction import background_model, foreground_mask, eigenback


def distorted_image_set(imgs, labels, num_runs=10):
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
	distorted_img_array = np.zeros([imgs.shape[0], imgs.shape[1], num_imgs * num_runs])
	for r in range(num_runs):
		start_index = r * num_imgs
		end_index = start_index + num_imgs
		distorted_img_array[:, :, start_index:end_index] = randomly_distort_images(imgs)
		print "done with run", r
	x = image_to_matrix(distorted_img_array)
	y = np.repeat(labels, num_runs)
	return x, y


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
		old_center = image_center(img) # (row, col)
		frame_center = np.array([[img.shape[0]/2], [img.shape[1]/2]])
		shift = np.round(np.subtract(frame_center, old_center))
		trans_tform = tf.SimilarityTransform(scale=1, translation=(shift[1], shift[0]))
		img = tf.warp(img, trans_tform.inverse)

		# 1. randomly select a rotation amount
		# randomly select a rotation angle between -40 to 41 degrees
		rotation_range = (np.arange(-30, 31, 1)/360) * 2 * math.pi
		rotation = random.choice(rotation_range)
		# rotate the image and save the result
		rotated_img = rotate_image(img, rotation=rotation)

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
			rotated_scaled_img = scale_image(rotated_img, scale)
		else:
			rotated_scaled_img = rotated_img

		# 3. randomly translate the image such that no cropping occurs
		# y_shift = reasonable translations along vertical axis
		row_total = rotated_scaled_img.sum(axis=1)
		row_active = np.where(row_total > 0)[0]
		y_shift_range = np.hstack((
			np.arange(-1, -1 * row_active.min(), -5),
			np.arange(1, img.shape[0] - row_active.max(), 5)
		))
		# x_shift = reasonable translations along horizontal axis
		col_total = rotated_scaled_img.sum(axis=0)
		col_active = np.where(col_total > 0)[0]
		x_shift_range = np.hstack((
			np.arange(-1, -1 * col_active.min(), -5),
			np.arange(1, img.shape[1] - col_active.max(), 5)
		))
		# randomly select a pair of reasonable shifts in either direction
		if len(y_shift_range) != 0:
			y_shift = random.choice(y_shift_range)
		else:
			y_shift = 0
		if len(x_shift_range) != 0:
			x_shift = random.choice(x_shift_range)
		else:
			x_shift = 0
		distorted_img = translate_image(rotated_scaled_img, (x_shift, y_shift))

		# save final results
		distorted_img_array[:, :, i] = distorted_img
	return distorted_img_array


def rotate_image(img, rotation):
	"""
	low level function to rotate a single image.
	This function finds the translation necessary to keep the image centered
	in order to keep the image from being cropped.
	:param img:
	:param rotation: counter-clockwise rotation in radians (e.g. math.pi / 4)
	:return:
	"""
	# need to pad the image before scaling it
	pad_width = 250
	padded_img = np.lib.pad(img, pad_width, pad_image)

	# define a rotation matrix
	rot_mat = np.array([
			[math.cos(rotation), math.sin(rotation) * -1],
			[math.sin(rotation), math.cos(rotation)]
		])
	# find how far the rotation will off-center the image:
	img_center = np.array([[padded_img.shape[0]/2], [padded_img.shape[1]/2]]) # (row, col)
	old_center = image_center(padded_img) # (row, col)
	rotated_center = rot_mat.T.dot(old_center)
	# translate the result so the center of mass is in center of image
	shift = np.round(np.subtract(rotated_center, img_center))
	# rotate then transform the image
	rot_tform = tf.SimilarityTransform(rotation=rotation)
	trans_tform = tf.SimilarityTransform(translation=(shift[0], shift[1]))
	padded_rotated_img = tf.warp(tf.warp(padded_img, rot_tform.inverse), trans_tform.inverse)
	# unpad the image
	return unpad_image(padded_rotated_img, pad_width)


def pad_image(img, pad_width, iaxis, kwargs):
	img[:pad_width[0]] = 0
	img[-pad_width[1]:] = 0
	return img


def unpad_image(img, pad_width):
	old_shape = (img.shape[0] - (2 * pad_width), img.shape[1] - (2 * pad_width))
	return img[pad_width:(pad_width + old_shape[0]), pad_width:(pad_width + old_shape[1])]


def scale_image(img, scale):
	"""
	Low level function to rescale a single image.
	This function also finds the necessary translation such that the image
	is still centered after being re-scaled
	:param img: an image of size x-pixels by y-pixels
	:param scale: an inverted number representing a scaling factor
		e.g. to increase size by 25%, scale = 1/1.25
	:return:
	"""
	# need to pad the image before scaling it
	pad_width = max(int(round((max(img.shape) * scale) - max(img.shape))), 250)
	padded_img = np.lib.pad(img, pad_width, pad_image)
	transformation = tf.SimilarityTransform(scale=scale)
	padded_scaled_img = tf.warp(padded_img, transformation.inverse)

	# next, determine how much to translate the image to keep it centered
	current_center = image_center(padded_scaled_img) # (row, col)
	desired_center = image_center(padded_img) # (row, col)
	shift = np.round(np.subtract(desired_center, current_center))
	# translate image to back to center
	transformation = tf.SimilarityTransform(translation=(shift[0], shift[1]))
	padded_centered_img = tf.warp(padded_scaled_img, transformation.inverse)
	# unpad the image back to its original dimensions and return it
	return unpad_image(padded_centered_img, pad_width)


def translate_image(img, translation):
	transformation = tf.SimilarityTransform(scale=1, translation=translation)
	return tf.warp(img, transformation.inverse)


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
