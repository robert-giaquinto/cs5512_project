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
import pickle


# 0. IMPORT DATA ----------------------------------------------------------------------
train_dir = '/Users/robert/documents/umn/5512_AI2/project/data/train/'
test_dir = '/Users/robert/documents/umn/5512_AI2/project/data/validation/'
train_set = np.loadtxt(train_dir + "distorted_train_set.csv", delimiter=',', dtype=np.float32)
train_labels = np.loadtxt(train_dir + "distorted_train_labels.csv", delimiter=',', dtype=np.float32)


# 1. CLUSTERING AND DIMENSION REDUCTION -----------------------------------------------
# use clustering class to reduce dimensionality and cluster the training set
do_pca = True
if do_pca:
	file_prefix = 'polynomial_'
	clustering = Cluster(n_clusters=20, n_components=10, method="KernelPCA")
	clustering = clustering.fit(train_set)
else:
	file_prefix = 'lda_'
	clustering = Cluster(n_clusters=20, n_components=10, method="LDA")
	clustering = clustering.fit(train_set, y=train_labels)
pickle.dump(clustering, open(file_prefix + "clustering.p", "wb"))
# clustering = pickle.load(open(file_prefix + "clustering.p", "rb"))

# apply methods to training
train_clusters = clustering.predict(train_set)
np.savetxt(train_dir + file_prefix + "reduc_dim_training.csv",
	clustering.dim_reduc.transform(train_set),
	fmt='%10.6f',
	delimiter=",")
del train_set

# load other ids, and prepare to put data into sliding window
train_seq_ids = np.loadtxt(train_dir + "distorted_train_seq_ids.csv", delimiter=',', dtype=np.int16)
train_frame_ids = np.loadtxt(train_dir + "distorted_train_frame_ids.csv", delimiter=',', dtype=np.int16)
train_ids = np.loadtxt(train_dir + "distorted_train_ids.csv", delimiter=',', dtype=np.int16)
# stack each variable into an array
training_frames = np.column_stack((train_labels, train_clusters, train_ids, train_seq_ids, train_frame_ids)).astype(np.uint16)
# save results before using sliding window
np.savetxt(train_dir + file_prefix + "non_windowed_training.csv", training_frames, fmt='%d', delimiter=",")


# transpose out_data into wide format
# use a sliding window to split each action sequence into a fixed number
# of frames
window_sz = 10
training_output = sliding_window(training_frames, window_sz)

# save clusters to csv file
np.savetxt(train_dir + file_prefix + "training.csv", training_output, fmt='%d', delimiter=",")
print "training saved"


# RUN ON TEST SET
x_test_mat = np.loadtxt(test_dir + "x_test_mat.csv", delimiter=',', dtype=np.float32)
np.savetxt(test_dir + file_prefix + "reduc_dim_testing.csv",
	clustering.dim_reduc.transform(x_test_mat),
	fmt='%10.6f',
	delimiter=",")

test_clusters = clustering.predict(x_test_mat)
print "clustering on test done"
del x_test_mat

y_test = np.loadtxt(test_dir + "y_test.csv", delimiter=',', dtype=np.int16)
test_seq_ids = np.loadtxt(test_dir + "distorted_test_seq_ids.csv", delimiter=',', dtype=np.int16)
test_frame_ids = np.loadtxt(test_dir + "distorted_test_frame_ids.csv", delimiter=',', dtype=np.int16)
test_ids = np.loadtxt(test_dir + "distorted_test_ids.csv", delimiter=',', dtype=np.int16)
testing_frames = np.column_stack((y_test, test_clusters, test_ids, test_seq_ids, test_frame_ids)).astype(np.uint16)
np.savetxt(test_dir + file_prefix + "non_windowed_testing.csv", testing_frames, fmt='%d', delimiter=",")


testing_output = sliding_window(testing_frames, window_sz)
np.savetxt(test_dir + file_prefix + "testing.csv", testing_output, fmt='%d', delimiter=",")
print "testing saved"
