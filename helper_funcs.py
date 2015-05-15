import numpy as np
from ChalearnLAPSample import ActionSample
import os
import cv2
import math
import scipy.misc as im


def load_all_videos(data_dir, drop_no_action=False, down_sample=None):
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
	# remove weird dot files, if exists
	file_names = [f for f in file_names if '._' not in f]

	# loop through files
	x = None
	y = None
	seq_ids = None
	frame_ids = None
	num_frames = 0  # store number of frames in each video
	video_lookup = {}
	action_lookup = {}
	for i, f in enumerate(file_names):
		seq = f[:-4]
		# load a video and convert to an frame matrix
		video_matrix, video_labels, action_labels = load_frame_matrix(f, data_dir, drop_no_action=drop_no_action, down_sample=down_sample)
		# save the video in a larger matrix
		if i > 0:
			x = np.vstack((x, video_matrix))
			y = np.hstack((y, video_labels))
			seq_ids = np.hstack((seq_ids, np.ones(video_matrix.shape[0]) * int(seq[-1])))
			frame_ids = np.hstack((frame_ids, (np.arange(video_matrix.shape[0]) + 1)))
		else:
			x = video_matrix
			y = video_labels
			seq_ids = np.ones(video_matrix.shape[0]) * int(seq[-1])
			frame_ids = np.arange(video_matrix.shape[0]) + 1

		# update the dictionary
		if num_frames == 0:
			num_frames += video_matrix.shape[0]
			video_lookup[seq] = [0, num_frames-1]
		else:
			video_lookup[seq] = [num_frames, num_frames + video_matrix.shape[0]-1]
			num_frames += video_matrix.shape[0]
		action_lookup[seq] = action_labels
	return x, y, video_lookup, action_lookup, seq_ids, frame_ids



def load_frame_matrix(file_name, data_dir, drop_no_action=False, down_sample=None):
	"""
	Converting everything to grayscale currently
	Only using frames the corresponding to an action
	:param file_names: a list of names of files to load (e.g. "Seq01.zip")
	:return: x, y
	"""
	# is this the test or training data
	# is_training = 'train' in data_dir
	# read in RGB images from using action sample function
	actionSample = ActionSample(data_dir + file_name)

	# read in ground truth seqXX_labels.csv
	# columns: ActorID, ActionID, StartFrame, EndFrame
	seq_labels = np.array(actionSample.getActions())
	num_frames = actionSample.getNumFrames()
	# num_frames = seq_labels[:,3].max()
	# if file_name == 'Seq02.zip':
	# 	num_frames = 840
	# elif file_name == 'Seq06.zip':
	# 	num_frames = 870
	# elif file_name == 'Seq08.zip':
	# 	num_frames = 960
	# elif file_name == 'Seq09.zip':
	# 	num_frames = 840

	# initialize  output data
	# store each image as a row
	sample_image = actionSample.getRGB(1)
	if down_sample is not None:
		sample_image = im.imresize(sample_image[:, :, 0], size=down_sample)
	print sample_image.shape

	x = np.zeros([num_frames, sample_image.shape[0]*sample_image.shape[1]])
	y = np.zeros(num_frames)
	# loop through each frame in the mp4 video
	for i in range(num_frames):
		# is this frame part of an action sequence
		there_is_a_label = np.any(np.logical_and(seq_labels[:, 2] <= i, seq_labels[:, 3] >= i))
		if there_is_a_label:
			# lookup which action this is
			label_row = np.where((seq_labels[:, 2] <= i) & (seq_labels[:, 3] >= i))[0]
			num_labels = len(label_row)
			if num_labels > 1:
				# multiple action occurring in this frame
				# choose action that occurs the most
				most_frames = 0
				for l in label_row:
					if (seq_labels[l, 3] - seq_labels[l, 2]) > most_frames:
						action = seq_labels[l, 1]
						most_frames = seq_labels[l, 3] - seq_labels[l, 2]
			else:
				action = seq_labels[label_row[0], 1]
			y[i] = action
		else:
			# assign action #12 to the observation
			y[i] = 12
		# load the image and convert it to a gray-scale image matrix
		img = actionSample.getRGB(i+1)
		gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
		if down_sample is not None:
			gray_img = im.imresize(gray_img, size=down_sample)
		x[i, :] = gray_img.flatten()

	if drop_no_action:
		keep_rows = np.where(y != 12)[0]
		x = x[keep_rows, :]
		y = y[keep_rows]
	return x, y, seq_labels


def matrix_to_image(mat, n_rows=90, n_cols=120):
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
			img_array[:, :, i] = mat[i, :].reshape(n_rows, n_cols)
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


def freq(x):
	y = np.bincount(x.astype(np.int64))
	ii = np.nonzero(y)[0]
	return zip(ii, y[ii])


def count_windows(action_freq, window_sz):
	num_obs = 0
	for f in action_freq:
		num_obs += max(f[1] - window_sz + 1, 1)
	return num_obs


# loop through actions and frames and put data in wide
# sliding window format
def sliding_window(training_frames, window_sz):
	# initialize output data:
	# how many frames make up each action?
	action_freq = freq(training_frames[:, 2])
	num_obs = count_windows(action_freq, window_sz)
	rval = np.zeros([num_obs, window_sz + 4], dtype=np.int16)

	# loop through each action
	frames_processed = 0
	out_index = 0
	for a in action_freq:
		num_frames = a[1]
		action = training_frames[frames_processed, 0]
		seq_id = training_frames[frames_processed, 3]
		if num_frames < window_sz:
			# need to repeat some of the frames
			start = frames_processed
			end = frames_processed + num_frames
			seq = training_frames[start:end, 1]
			times_repeated = math.floor(window_sz / num_frames)
			remainder = window_sz % num_frames
			if remainder == 0:
				cluster_sequence = np.tile(seq, times_repeated)
			else:
				cluster_sequence = np.hstack((np.tile(seq, times_repeated), seq[0:remainder]))

			# stack results into an array
			frame_id_start = training_frames[start, 4]
			frame_id_end = training_frames[end-1, 4]  # end was defined non-inclusively, so minus 1
			rval[out_index, :] = np.hstack((action, cluster_sequence, seq_id, frame_id_start, frame_id_end))
			out_index += 1
		else:
			# set up sliding window
			num_windows = num_frames - window_sz + 1
			for w in range(num_windows):
				# what is start and end index of current window?
				start = frames_processed + w
				end = frames_processed + w + window_sz
				cluster_sequence = training_frames[start:end, 1]

				# stack results into an array
				frame_id_start = training_frames[start, 4]
				frame_id_end = training_frames[end-1, 4]  # end was defined non-inclusively, so minus 1
				rval[out_index, :] = np.hstack((action, cluster_sequence, seq_id, frame_id_start, frame_id_end))
				out_index += 1
		frames_processed += num_frames
	return rval.astype(np.float32)


