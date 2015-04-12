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
import time
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


# import data
data_dir = '/Users/robert/documents/umn/5512_AI2/project/data/train/'
# data_dir = 'C:\\Users\\VAIO\\Desktop\\Spring 2015\\AI2\\Project\\code\\data\\'
file_names = [f for f in os.listdir(data_dir) if f.endswith('.zip')]
x, y, video_lookup = load_all_videos(data_dir)
# split video into training and test
x_train = x[0:1956, :]
x_test = x[1957:3092, :]
y_train = y[0:1956, :]
y_test = y[1957:3092, :]


# (using walk and run actions as training the background model)
background_train = x_train[ np.where((y_train == 6) | (y_train == 7))[0], ]


# find background image
back_vec = background_model(background_train) # takes about 10 sec
# subtract background from each image, return an image array
img_array = eigenback(back_vec, x_train, back_thres=.25, fore_thres=.1, rval='mask_array', blur=True) # takes about 30 sec


# create a dataset of distorted images
tstart = time.time()
training_set, training_labels = distorted_image_set(img_array, y_train, 3)
tend = time.time()
print "Time to process", training_set.shape[0], "images =", round(tend - tstart, 3)
# save training and labels to disk
np.savetxt("training_set.csv", training_set, fmt='%10.5f', delimiter=",")
np.savetxt("training_labels.csv", training_labels, fmt='%d', delimiter=",")
# import csv's
training_set = np.genfromtxt("training_set.csv", delimiter=',')
training_labels = np.genfromtxt("training_labels.csv", delimiter=',')


# use clustering class to reduce dimensionality and cluster the training set
cluster = Cluster(n_clusters=60, n_components=50)
cluster = cluster.fit(training_set)
training_clusters = cluster.predict(training_set)
print "Total explained variance:", sum(cluster.pca.explained_variance_ratio_)
output = np.vstack((training_labels, training_clusters)).T
# save clusters to csv file
np.savetxt("clustered_training_set.csv", output, fmt='%10.5f', delimiter=",")
# apply methods to test set
test_clusters = cluster.predict(x_test)
output = np.vstack((y_test.reshape(len(y_test)), test_clusters)).T
np.savetxt("clustered_test_set.csv", output, fmt='%10.5f', delimiter=",")









# what is the frequency of each cluster and true action?
names = ('wave', 'point', 'clap', 'crouch', 'jump', 'walk', 'run', 'shake hands', 'hug', 'kiss', 'fight')
y_labels = []
for i in range(len(training_labels)):
	y_labels.append(names[int(training_labels[i])-1])
y_labels = np.array(y_labels)

import pandas as pd
bothy = np.vstack((training_clusters, y_labels)).T
yp = pd.DataFrame(bothy, columns=['y_test','label'])
freq = pd.pivot_table(yp, rows='y_test', cols='label', aggfunc=len, fill_value=0)
pct = np.multiply(freq, (1 / freq.sum(axis=1)).reshape((freq.shape[0], 1)))
pct.to_csv("clusters_and_labels.csv")