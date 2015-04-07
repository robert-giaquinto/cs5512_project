from __future__ import division
import math
import random
import numpy as np
from skimage import transform as tf
from background_subtraction import background_model, foreground_mask, eigenback


# TODO: write wrapper function to build out the artifical training data set using functions below.
def distorted_image_set(x_train):
	print "TBD"
	return distorted_images


def randomly_rotate_images(img_array):
	"""
	This function takes an image array and, for each image, the image is rotated
	by some random amount between 45 and -45 degrees,
	each image is then returned as part of an imaged array
	:param img_array: a 3D array of images
	:return: a 3D image array where each image from input has been rotated by
		some reasonable amount.
	"""
	# loop through each image and rotate it
	num_frames = img_array.shape[2]
	rotated_img_array = np.zeros(img_array.shape)
	for i in range(num_frames):
		img = img_array[:, :, i]

		# randomly select a rotation angle betwen -45 to 45 degrees
		rotation_range = (np.hstack((np.arange(-45, -4, 1), np.arange(5, 46, 1)))/360) * 2 * math.pi
		rotation = random.choice(rotation_range)

		# rotate the image and save the restult
		rotated_img = rotate_image(img, rotation=rotation)
		rotated_img_array[:, :, i] = rotated_img

	return rotated_img_array


def rotate_image(img, rotation):
	"""
	low level function to rotate a single image.
	This function finds the translation necessary to keep the image centered
	in order to keep the image from being cropped.
	:param img:
	:param rotation: counter-clockwise rotation in radians (e.g. math.pi / 4)
	:return:
	"""
	# define a rotation matrix
	rot_mat = np.array([
			[math.cos(rotation), math.sin(rotation) * -1],
			[math.sin(rotation), math.cos(rotation)]
		])
	# find how far the rotation will off-center the image:
	old_center = image_center(img)
	new_center = rot_mat.dot(old_center)
	shift = np.round(np.subtract(old_center, new_center))
	# transform the image
	tform = tf.SimilarityTransform(scale=1, rotation=rotation, translation=(shift[0], shift[1]))
	# another option that doesn't require manual translation:
	#from scipy.ndimage.interpolation import rotate
	#rotate(img, angle=30, reshape=True)
	return tf.warp(img, tform)


def randomly_scale_images(img_array):
	"""
	This function takes an image array and, for each image, finds the most that it
	could be enlarged without cropping the image. The image is then re-scaled
	by some random amount and returned as part of an imaged array
	:param img_array:
	:return: an image array, where each element corresponds to an image from input
		image array -- except each image has been re-scaled by a random amount
	"""
	# loop through each image and re-scale it
	num_frames = img_array.shape[2]
	scaled_img_array = np.zeros(img_array.shape)
	for i in range(num_frames):
		img = img_array[:, :, i]

		# find max amount that the image can be scaled without cropping
		# sum across rows to find columns active in foreground
		y_total = img.sum(axis=0)
		y_active = np.where(y_total > 0)[0]
		y_len = y_active.max() - y_active.min()
		# sum across columns to find row active in foreground
		x_total = img.sum(axis=1)
		x_active = np.where(x_total > 0)[0]
		x_len = x_active.max() - x_active.min()
		# for some reason the scaling factor needs to be inverted (i.e. 1 / scale)
		scale_min = round(1. / (math.floor(min(img.shape[0]/x_len, img.shape[1]/y_len) * 10) / 10), 2)
		scale_range = np.hstack((np.arange(scale_min, .96, .01), np.arange(1.05, 2 - scale_min + .01, .01)))
		scale = random.choice(scale_range)

		# scale and save image
		scaled = scale_image(img, scale)
		scaled_img_array[:, :, i] = scaled
	return scaled_img_array


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
	# convert image to a mask in order to find center
	old_center = image_center(img)
	new_center = np.round(old_center / scale)
	shift = np.round(np.subtract(new_center, old_center)).flatten()
	transformation = tf.SimilarityTransform(scale=scale, translation=(shift[0], shift[1]))
	return tf.warp(img, transformation)


def image_center(img):
	"""
	find the x,y coordinates of the weighted center of an image
	this is helpful for re-centering images that have been rotated
	or stretched
	:param img: a foreground image, dimensions = x-pixels by y-pixels
	:return:
	"""
	# what percent of pixels below to each row?
	foreground_size = np.count_nonzero(img)
	y_weight = (img != 0).sum(axis=1) / foreground_size
	# multiply weight element-wise by the position of the pixel
	y_weighted_position = np.multiply(y_weight, range(len(y_weight)))
	# at what y-position are half of the pixels observed?
	y_center_of_mass = np.where(y_weighted_position.cumsum() > y_weighted_position.sum()/2)[0][0]
	# repeat for x-value
	x_weight = (img != 0).sum(axis=0) / foreground_size
	x_weighted_position = np.multiply(x_weight, range(len(x_weight)))
	x_center_of_mass = np.where(x_weighted_position.cumsum() > x_weighted_position.sum()/2)[0][0]
	return np.array([[x_center_of_mass], [y_center_of_mass]])
