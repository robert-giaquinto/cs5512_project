import numpy as np
from ChalearnLAPSample import ActionSample
import os
import cv2
import random


def load_all_videos(data_dir):
	"""
	Load all videos stored in a folder and the labels associated with each video
	:param data_dir:
	:return:
		x - a matrix of frames (rows) and pixels (cols) for each video
		y - a vector with labels corresponding to each row of x
		video_lookup - a dictionary whose key is the file that was loaded
			and whose value is a list of two integers that corresponds to the first and
			last row in x that is a frame from the file.
			For example, {Seq01: [0, 4], Seq02: [5, 9]} implies the video seq01 is in frames
			0 through 4, and seq02 is in frames 5 through 9
	"""
	# only look at zip files in the directory
	file_names = sorted([f for f in os.listdir(data_dir) if f.endswith('.zip')])

	# loop through files
	x = None
	y = None
	num_frames = 0 # store number of frames in each video
	video_lookup = {}
	action_lookup = {}
	for f in file_names:
		seq = f[:-4]
		# load a video and convert to an frame matrix
		video_matrix, video_labels, action_labels = load_frame_matrix(f, data_dir)
		# save the video in a larger matrix
		try:
			x = np.vstack((x, video_matrix))
			y = np.vstack((y, video_labels))
		except:
			x = video_matrix
			y = video_labels
		# update the dictionary
		if num_frames == 0:
			num_frames += video_matrix.shape[0]
			video_lookup[seq] = [0, num_frames-1]
		else:
			video_lookup[seq] = [num_frames, num_frames + video_matrix.shape[0]-1]
			num_frames += video_matrix.shape[0]
		action_lookup[seq] = action_labels
	return x, y, video_lookup, action_lookup



def load_frame_matrix(file_name, data_dir):
	"""
	Converting everything to grayscale currently
	Only using frames the corresponding to an action
	:param file_names: a list of names of files to load (e.g. "Seq01.zip")
	:return: x, y
	"""
	# read in RGB images from using action sample function
	actionSample = ActionSample(data_dir + file_name)

	# read in ground truth seqXX_labels.csv
	# columns: ActorID, ActionID, StartFrame, EndFrame
	seq_labels = np.array(actionSample.getActions())

	# loop through each frame in the mp4 video
	num_frames = actionSample.getNumFrames()

	# initialize data
	# store each image as a row
	# ignoring the actor in the labels dataset
	sample_image = actionSample.getRGB(1)
	all_x = np.zeros([num_frames, sample_image.shape[0]*sample_image.shape[1]])
	all_y = np.zeros(num_frames)
	for i in range(num_frames):
		# is this frame part of an action sequence
		there_is_a_label = np.any(np.logical_and(seq_labels[:, 2] <= i, seq_labels[:, 3] >= i))
		if there_is_a_label:
			# lookup which action this is
			label_row = np.where((seq_labels[:, 2] <= i) & (seq_labels[:, 3] >= i))[0][0]
			all_y[i] = seq_labels[label_row, 1]
			# load the image and convert it to a gray-scale image matrix
			img = actionSample.getRGB(i+1)
			gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
			all_x[i, :] = gray_img.flatten()
	# TODO: currently storing all images and dropping the ones without a label, can be more efficient
	keep_rows = np.where(all_y != 0)[0]
	x = all_x[keep_rows, :]
	y = all_y[keep_rows]
	return x, y.reshape((len(y), 1)), seq_labels


def matrix_to_image(mat, n_rows=360, n_cols=480):
	"""
	Take an image frame matrix and convert it back into a image array
	for example, take 2d matrix, convert to 3d array of images

	:param mat:
	:param n_cols: number of columns in the original image
	:return:
	"""
	if len(mat.shape) > 1:
		num_frames = mat.shape[0]
		img_array = np.zeros([n_rows, n_cols, num_frames])
		for i in range(num_frames):
			img_array[:, :, i] = mat[i,:].reshape(n_rows, n_cols)
	else:
		img_array = mat.reshape(n_rows, n_cols)
	return img_array


def image_to_matrix(img_array):
	"""
	Convert a 3D image array to a 2D matrix
	:param img_array:
	:return:
	"""
	num_frames = img_array.shape[2]
	rval = np.zeros([num_frames, img_array.shape[0]*img_array.shape[1]])
	for i in range(num_frames):
		rval[i, :] = img_array[:, :, i].flatten()
	return rval



# create an action id for each row of x
def label_action_ids(y, n=1):
	ids = np.ones(len(y) * n)
	for n in range(n):
		for i in range(len(y)):
			ind = i + (n * len(y))
			if i == 0 and n == 0:
				ids[ind] = 1
			elif i == 0 and n > 0:
				# starting distortions, so increment id
				ids[ind] = ids[ind-1] + 1
			else:
				if y[i] == y[i-1]:
					# unchanged action, so keep same id
					ids[ind] = ids[ind-1]
				else:
					# new action appeared, increment id
					ids[ind] = ids[ind-1] + 1
	return ids


# compute the max number of frames per action
def get_max_frames(action_lookup):
	max_frames = 0
	# loop through each video
	for k in action_lookup.keys():
		action_dict = action_lookup[k]
		# loop through each action
		for a in range(action_dict.shape[0]):
			num_frames = action_dict[a, 3] - action_dict[a, 2]
			max_frames = max(num_frames, max_frames)
	return max_frames