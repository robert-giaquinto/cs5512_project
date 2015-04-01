from __future__ import division
import numpy as np
from ChalearnLAPSample import ActionSample
import os
import cv2
import math
from helper_funcs import *
from background_subtraction import *

import scipy.stats as ss
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
# NO NEED TO RUN, IT TAKES A WHILE
# fore_mat, fore_mask = eigenback(x_train, x, back_thres=.15, method='mean', n_components=25, fore_thres=.25)


# apply unsupervised clustering on each foreground image in x matrix.
back_vec = background_model(background_train)
cluster = Cluster(back_vec)
cluster = cluster.fit(x_train)
# cluster_labels = cluster.predict(x_test)
cluster_labels = cluster.predict(x_train)

test = cluster.cluster.cluster_centers_
a_cluster = test[10, :]
test2 = cluster.pca.inverse_transform(a_cluster)
img = matrix_to_image(test2)
img = img - img.min()
img = img * 255 / img.max()
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

