from __future__ import division
import numpy as np
from ChalearnLAPSample import ActionSample
import os
import cv2
import math
from helper_funcs import *
from background_subtraction import *
from image_distortion import *


# 0. IMPORT DATA ----------------------------------------------------------------------
train_dir = '/Users/robert/documents/umn/5512_AI2/project/data/train/'
test_dir = '/Users/robert/documents/umn/5512_AI2/project/data/validation/'
# train_dir = '/home/smit7982/Documents/cs5512/data/train/'
# test_dir = '/home/smit7982/Documents/cs5512/data/validation/'
x_train, y_train, video_lookup, action_lookup, train_seq_ids, train_frame_ids = load_all_videos(train_dir, drop_no_action=True, down_sample=0.25)
num_rows = x_train.shape[0]
print "input data (" + str(num_rows) + ", " + str(x_train.shape[1]) + ")"



# 1. SUBTRACT BACKGROUND ----------------------------------------------------------
# find background image
# using walk and run actions as training the background model
back_vec = background_model(x_train[ np.where((y_train == 6) | (y_train == 7))[0], ], n_components=5)

# subtract background from each image, return an image array
x_train_array = eigenback(back_vec, x_train, back_thres=.2, fore_thres=.1, rval='mask_array', blur=True)
print x_train_array.shape



# 2. DISTORT AND INCREASE DATASET SIZE ---------------------------------------------
# create a dataset of distorted images, save them to disk
num_distortions = 6
train_set = distorted_image_set(x_train_array, num_distortions)
train_labels = np.tile(y_train, num_distortions+1)
train_seq_ids = np.tile(train_seq_ids, num_distortions+1)
train_frame_ids = np.tile(train_frame_ids, num_distortions+1)

# give new distorted images an id
train_ids = label_action_ids(y_train, num_distortions+1)

# remove unnecessary items in memory
del x_train, x_train_array, y_train

# save results for other programs
np.savetxt(train_dir + "distorted_train_set.csv", train_set, fmt='%d', delimiter=",")
np.savetxt(train_dir + "distorted_train_labels.csv", train_labels, fmt='%d', delimiter=",")
np.savetxt(train_dir + "distorted_train_seq_ids.csv", train_seq_ids, fmt='%d', delimiter=",")
np.savetxt(train_dir + "distorted_train_frame_ids.csv", train_frame_ids, fmt='%d', delimiter=",")
np.savetxt(train_dir + "distorted_train_ids.csv", train_ids, fmt='%d', delimiter=",")
del train_seq_ids, train_labels, train_set, train_frame_ids, train_ids



# 3. APPLY RESULTS TO TEST DATA ------------------------------------------------------
x_test, y_test, video_lookup_test, action_lookup_test, test_seq_ids, test_frame_ids = load_all_videos(test_dir, down_sample=0.25)
print x_test.shape
test_ids = label_action_ids(y_test)

x_test_mat = eigenback(back_vec, x_test, back_thres=.2, fore_thres=.1, rval='mask_mat', blur=True)
print x_test_mat.shape

np.savetxt(test_dir + "x_test_mat.csv", x_test_mat, fmt='%d', delimiter=",")
np.savetxt(test_dir + "y_test.csv", y_test, fmt='%d', delimiter=",")
np.savetxt(test_dir + "distorted_test_seq_ids.csv", test_seq_ids, fmt='%d', delimiter=",")
np.savetxt(test_dir + "distorted_test_frame_ids.csv", test_frame_ids, fmt='%d', delimiter=",")
np.savetxt(test_dir + "distorted_test_ids.csv", test_ids, fmt='%d', delimiter=",")
