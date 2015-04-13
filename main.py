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


# import data
data_dir = '/Users/robert/documents/umn/5512_AI2/project/data/train/'
# data_dir = 'C:\\Users\\VAIO\\Desktop\\Spring 2015\\AI2\\Project\\code\\data\\'
file_names = [f for f in os.listdir(data_dir) if f.endswith('.zip')]
x, y, video_lookup = load_all_videos(data_dir)
# split video into training and test
# x_train = x[0:1956, :].astype(np.float16)
# x_test = x[1957:3092, :].astype(np.float16)
# y_train = y[0:1956, :].astype(np.float16)
# y_test = y[1957:3092, :].astype(np.float16)

# DEVELOPMENT: LIMIT DATA SIZE
x_train = x[ np.where((y == 1) | (y == 2))[0], ]
y_train = y[ np.where((y == 1) | (y == 2))[0], ]
x_train.shape


# (using walk and run actions as training the background model)
background_train = x_train[ np.where((y_train == 6) | (y_train == 7))[0], ]
background_train = x[ np.where((y == 6) | (y == 7))[0], ]
background_train.shape

# find background image
back_vec = background_model(background_train) # takes about 10 sec
# subtract background from each image, return an image array
img_array = eigenback(back_vec, x_train, back_thres=.25, fore_thres=.1, rval='mask_array', blur=True) # takes about 30 sec


# create a dataset of distorted images, save them to disk
tstart = time.time()
distorted_image_set(img_array, y_train, 2, data_dir=data_dir)
# training_set, training_labels = distorted_image_set(img_array, y_train, 10)
tend = time.time()
print "Time to process images =", round(tend - tstart, 3)

# import csv's
training_set = np.loadtxt(data_dir + "distorted_matrix.csv", delimiter=',', dtype=np.float16)
training_labels = np.repeat(y_train, 2)
del x, y


# use clustering class to reduce dimensionality and cluster the training set
cluster = Cluster(n_clusters=16, n_components=25, n_jobs=1)
cluster = cluster.fit(training_set)
print "Total explained variance:", sum(cluster.dim_reduc.explained_variance_ratio_)

cluster = Cluster(n_clusters=16, n_components=25, n_jobs=1, method="IncrementalPCA")
cluster = cluster.fit(training_set)
print "Total explained variance:", sum(cluster.dim_reduc.explained_variance_ratio_)

cluster = Cluster(n_clusters=16, n_components=25, n_jobs=1, method="KernelPCA")
cluster = cluster.fit(training_set)

cluster = Cluster(n_clusters=16, n_components=25, n_jobs=1, method="LDA")
cluster = cluster.fit(training_set, training_labels)


# apply methods to trainig/test set
training_clusters = cluster.predict(training_set)
output = np.vstack((training_labels, training_clusters)).T.astype(np.uint8)
# save clusters to csv file
np.savetxt(data_dir + "kernelPCA_training.csv", output, fmt='%d', delimiter=",")




# TODO: SELECT images that are closest to the centroids, plot those.
# plot some of the clusters
cluster_centers = cluster.cluster.cluster_centers_
plt.figure(figsize=(20,20))
for c in range(16):
	a_cluster = cluster_centers[c, :]
	img_matrix = cluster.dim_reduc.inverse_transform(a_cluster)
	img = matrix_to_image(img_matrix)[:,:,0]
	img = img - img.min()
	img = img * 255 / img.max()
	fig_num = c + 1
	plt.subplot(4,4,fig_num)
	plt.axis("off")
	plt.imshow(img.astype('uint8'), cmap='Greys_r')
	plt.title('cluster' + str(c))
plt.suptitle('LDA')
plt.show()




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
# pct.to_csv(data_dir + "clusters_and_labels.csv")
