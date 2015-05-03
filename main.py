from __future__ import division
import numpy as np
from ChalearnLAPSample import ActionSample
import os
import cv2
import math
from helper_funcs import *
from background_subtraction import *
from image_distortion import *
from cluster_images import Cluster


# import data
data_dir = '/Users/robert/documents/umn/5512_AI2/project/data/train/'
# data_dir = 'C:\\Users\\VAIO\\Desktop\\Spring 2015\\AI2\\Project\\code\\data\\'
file_names = [f for f in os.listdir(data_dir) if f.endswith('.zip')]
x, y, video_lookup, action_lookup = load_all_videos(data_dir)


# DEVELOPMENT: LIMIT DATA SIZE
x_train = x[ np.where((y == 1) | (y == 2))[0], ]
y_train = y[ np.where((y == 1) | (y == 2))[0], ]
x_train.shape

# split video into training and test
# x_train = x[0:1956, :].astype(np.float16)
# x_test = x[1957:3092, :].astype(np.float16)
# y_train = y[0:1956, :].astype(np.float16)
# y_test = y[1957:3092, :].astype(np.float16)


# set global parameters based on video meta data
num_actions = 11
max_frames = get_max_frames(action_lookup)




# (using walk and run actions as training the background model)
background_train = x_train[ np.where((y_train == 6) | (y_train == 7))[0], ]
background_train = x[ np.where((y == 6) | (y == 7))[0], ]
background_train.shape

# find background image
back_vec = background_model(background_train)
# subtract background from each image, return an image array
img_array = eigenback(back_vec, x_train, back_thres=.25, fore_thres=.1, rval='mask_array', blur=True)
img_array.shape

# create a dataset of distorted images, save them to disk
num_distortions = 2
distorted_image_set(img_array, y_train, num_distortions, data_dir=data_dir)

# import csv's
training_set = np.loadtxt(data_dir + "distorted_matrix.csv", delimiter=',', dtype=np.float16)
training_labels = np.tile(y_train, [num_distortions, 1])
# del x, y
# give new distorted images a new id
training_ids = label_action_ids(y_train, num_distortions)



# use clustering class to reduce dimensionality and cluster the training set
clustering = Cluster(n_clusters=16, n_components=25, n_jobs=1, method="KernelPCA")
clustering = clustering.fit(training_set)

# apply methods to training/test set
training_clusters = clustering.predict(training_set)
training_frames = np.vstack((training_labels.reshape(710), training_clusters, training_ids)).T.astype(np.uint8)

# transpose out_data into wide format
# use a sliding window to split each action sequence into a fixed number
# of frames
window_sz = 10


def freq(x):
	y = np.bincount(x)
	ii = np.nonzero(y)[0]
	return zip(ii, y[ii])


def count_windows(action_freq, window_sz):
	num_obs = 0
	for f in action_freq:
		num_obs += f[1] - window_sz + 1
	return num_obs


# loop through actions and frames and put data in wide
# sliding window format
def sliding_window(training_frames, window_sz):
	# initialize output data:
	# how many frames make up each action?
	action_freq = freq(training_frames[:,2])
	num_obs = count_windows(action_freq, window_sz)
	rval = np.zeros([num_obs, window_sz+1])

	# loop through each action
	frames_processed = 0
	out_index = 0
	for a in action_freq:
		num_frames = a[1]
		action = training_frames[frames_processed, 0]
		if num_frames < window_sz:
			# need to repeat some of the frames
			start = frames_processed
			end = frames_processed + num_frames
			seq = training_frames[start:end, 1]
			times_repeated = math.floor(window_sz / num_frames)
			remainder = window_sz % num_frames
			cluster_sequence = np.hstack((np.tile(seq, times_repeated), seq[0:remainder]))
			rval[out_index, :] = np.hstack((action, cluster_sequence))
			out_index += 1
		else:
			# set up sliding window
			num_windows = num_frames - window_sz + 1
			for w in range(num_windows):
				# what is start and end index of current window?
				start = frames_processed + w
				end = frames_processed + w + window_sz
				rval[out_index, :] = np.hstack((action, training_frames[start:end, 1]))
				out_index += 1
		frames_processed += num_frames
	return rval

output = sliding_window(training_frames, window_sz)

# save clusters to csv file
np.savetxt(data_dir + "kernelPCA_training.csv", output, fmt='%d', delimiter=",")



