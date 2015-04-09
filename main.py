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

import matplotlib.pyplot as plt




# import data
data_dir = '/Users/robert/documents/umn/5512_AI2/project/data/train/'
# data_dir = 'C:\\Users\\VAIO\\Desktop\\Spring 2015\\AI2\\Project\\code\\data\\'
file_names = [f for f in os.listdir(data_dir) if f.endswith('.zip')]

x, y, video_lookup = load_all_videos(data_dir)
x_train = x[0:1956, :]
x_test = x[1957:3092, :]
y_train = y[0:1956, :]
y_test = y[1957:3092, :]

# run analysis on just one file for now
# file_name = file_names[0]
# load in one video
# x, y = load_frame_matrix(file_name, data_dir)


# split video into training and test
# (using walk and run actions as training the background model)
background_train = x_train[ np.where((y_train == 6) | (y_train == 7))[0], ]


# subtract background from each image
# train algo on x_train, but apply it to each frame in x
back_vec = background_model(background_train) # takes about 10 sec
img_array = eigenback(back_vec, x_train, back_thres=.25, fore_thres=.1, rval='mask_array', blur=True) # takes about 30 sec
# create a dataset of distorted images
import time
tstart = time.time()
training_set = randomly_distort_images(img_array)
tend = time.time()
print "Time to process", img_array.shape[2], "images =", round(tend - tstart, 3)

fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(ncols=3, nrows=2)
ax1.imshow(training_set[:,:, 0])
ax2.imshow(training_set[:,:, 1])
ax3.imshow(training_set[:,:, 2])
ax4.imshow(training_set[:,:, 3])
ax5.imshow(training_set[:,:, 4])
ax6.imshow(training_set[:,:, 5])
plt.show()

# cluster the image distortions
mask_mat = image_to_matrix(training_set)
pca = RandomizedPCA(n_components=50).fit(mask_mat)
print "Total explained variance:", sum(pca.explained_variance_ratio_)
fore_pca = pca.transform(mask_mat)

# apply unsupervised clustering on each image.
cluster = KMeans(n_clusters=30).fit(fore_pca)
cluster_labels = cluster.predict(fore_pca)






# OLD:


# apply unsupervised clustering on each foreground image in x matrix.
# back_vec = background_model(background_train, method='mean', n_components=10)
# cluster = Cluster(back_vec, n_clusters=30, n_components=50)
# cluster = cluster.fit(x_train)
# # cluster_labels = cluster.predict(x_test)
# cluster_labels = cluster.predict(x_train)

cluster_centers = cluster.cluster_centers_
plt.figure(figsize=(20,20))
for c in range(25):
	a_cluster = cluster_centers[c, :]
	img_matrix = pca.inverse_transform(a_cluster)
	img = matrix_to_image(img_matrix)
	img = img - img.min()
	img = img * 255 / img.max()
	fig_num = c + 1
	plt.subplot(5,5,fig_num)
	plt.axis("off")
	plt.imshow(img.astype('uint8'), cmap='Greys_r')
plt.show()




# what is the frequency of each cluster and true action?
names = ('wave', 'point', 'clap', 'crouch', 'jump', 'walk', 'run', 'shake hands', 'hug', 'kiss', 'fight')
y_labels = []
for i in range(len(y_train)):
	y_labels.append(names[int(y_train[i])-1])
y_labels = np.array(y_labels)

import pandas as pd
bothy = np.vstack((cluster_labels, y_labels)).T
yp = pd.DataFrame(bothy, columns=['y_test','label'])
freq = pd.pivot_table(yp, rows='y_test', cols='label', aggfunc=len, fill_value=0)
pct = np.multiply(freq, (1 / freq.sum(axis=1)).reshape((freq.shape[0], 1)))

