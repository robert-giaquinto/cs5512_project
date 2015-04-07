from __future__ import division
import numpy as np
from skimage import feature

# I put this is a separate file because it doesn't appear to be very usefule
# can include this later if needed.
def detect_edges(mask_array, sigma=3):
	"""
	detect edges on a binary image array
	:param mask_array: a 3D image array of foreground masks
	:return:
	"""
	num_images = mask_array.shape[2]
	edge_array = np.zeros(mask_array.shape)
	for i in range(num_images):
		edge_array[:, :, i] = feature.canny(mask_array[:,:, i], sigma=sigma)
	return edge_array