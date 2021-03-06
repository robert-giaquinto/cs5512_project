from __future__ import division
import numpy as np
from ChalearnLAPSample import ActionSample
import os
import cv2
import math
from helper_funcs import *
from background_subtraction import *

# import data
data_dir = '/Users/robert/documents/umn/5512_AI2/project/data/train/'
data_dir = 'C:\\Users\\VAIO\\Desktop\\Spring 2015\\AI2\\Project\\code\\data\\'
file_names = [f for f in os.listdir(data_dir) if f.endswith('.zip')]

# load in all videos (takes about 15 sec per video
# x, y, video_lookup = load_all_videos(data_dir)
# print video_lookup

# load in one video
x, y = load_frame_matrix(file_names[0], data_dir)

# split video into training and test
# use the run and walk videos to train background subtraction
train_rows = np.where((y == 6) | (y == 7))[0]
test_rows = np.where((y != 6) & (y != 7))[0]
x_train = x[train_rows, ]
x_test = x[test_rows, ]
y_train = y[train_rows, ]
y_test = y[test_rows, ]

# create the background image model
back_vec = background_model(x_train)
# for fun see what the background image looks like
# transform background matrix into an image frame
back_img = matrix_to_image(back_vec, 360, 480)
cv2.imshow('pic', back_img.astype('uint8'))
cv2.waitKey(1)
cv2.destroyAllWindows()

# use eigenback algorithm to remove background from images
fore_mask = eigenback(back_vec, x_test[0])
fore_img = matrix_to_image(fore_mask, 360, 480) * 255
cv2.imshow('pic', fore_img.astype('uint8'))
cv2.waitKey(1)
cv2.destroyAllWindows()

# for fun: use mask to only show foreground
fore_matrices = np.multiply(fore_mask, x_test)
fore_imgs = matrix_to_image(fore_matrices, 360, 480)
# pull out one image, convert it to integer encoding
img = fore_imgs[:, :, 0].astype('uint8')
cv2.imshow('pic', img)
cv2.waitKey(1)
cv2.destroyAllWindows()
# write to file for sharing
cv2.imwrite('foreground.png', img)
