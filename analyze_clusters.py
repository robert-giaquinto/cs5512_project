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

file_prefix = 'polynomial_'
clustering = pickle.load(open(file_prefix + "clustering.p", "rb"))
train_clusters = clustering.predict(train_set)

imgs = train_set[0:(train_set.shape[0]/6), :]
labs = train_clusters[0:(train_set.shape[0]/6)]
imgs_reduc = clustering.dim_reduc.transform(imgs)

from matplotlib import pyplot as plt


topx = 9
for i, c in enumerate(np.unique(train_clusters)):
	cluster_reduc = imgs_reduc[np.where(labs == c)[0]]
	if cluster_reduc.shape[0] == 0:
		print "skipping cluster", c
		continue
	center = clustering.cluster.cluster_centers_[i]
	# find distance of observation to cluster center
	dist = np.linalg.norm(np.subtract(cluster_reduc, center), axis=1)

	# pull out original images closest to the centroids
	a_cluster = imgs[np.where(labs == c)[0]]
	best_imgs = a_cluster[np.argsort(dist),:]
	img = matrix_to_image(best_imgs[0:topx, :])

	# plot images, and save
	plt.figure(i)
	plt.suptitle('Images Nearest Cluster ' + str(c))
	for j in range(min(topx, img.shape[2])):
		plt.subplot(3, 3, j+1)
		plt.imshow(img[:,:,j], cmap="gray")
		plt.axis('off')
	plt.savefig(train_dir + 'cluster' + str(c) + '.png')






# TODO: SELECT images that are closest to the centroids, plot those.
# plot some of the clusters
cluster_centers = clustering.cluster.cluster_centers_
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



plt.figure(1)
plt.subplot(1,2,1)
plt.imshow(img)
plt.subplot(1,2,2)
plt.imshow(rotated_scaled_img)
plt.show()