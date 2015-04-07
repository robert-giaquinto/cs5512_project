from __future__ import division
from sklearn.cluster import KMeans
from background_subtraction import background_model, foreground_mask, eigenback
from image_distortion import distorted_image_set
from helper_funcs import matrix_to_image, image_to_matrix

class Cluster(object):
	"""
	TBD
	Only doing Kmeans clustering right now
	Before clustering each image has:
		1. background subtracted
		2. only contiguous regions of foreground are kept
		skip: ---# 3. edges of foreground are extracted
		4. foreground is mapped to a lower dimension using PCA
		5. finally, the images are clustered
	"""
	def __init__(self, back_vec,
			n_clusters=30,
			n_components=50,
			back_thres=.25,
			fore_thres=.1,
			n_jobs=1):
		# must have parameters:
		self.back_vec = back_vec

		# optional parameters
		self.n_clusters = n_clusters
		self.n_components = n_components
		self.back_thres = back_thres
		self.fore_thres = fore_thres
		self.n_jobs = n_jobs

		# to be assigned in as later
		self.cluster = None
		self.pca = None

	def fit(self, x):
		"""
		:param x: an image matrix to train a clustering algorithm on
		"""
		# use the background image given to extract foreground from x
		tstart = time.time()
		mask_mat = eigenback(self.back_vec, x, self.back_thres, self.fore_thres, rval='mask_mat', blur=True)
		# tend = time.time()
		# print "foreground mask extracted in", tend - tstart, "seconds."
		# TOO SLOW, TRY SOMETHING SIMPLER:
		# mask_mat = foreground_mask(self.back_vec, x, self.back_thres)
		tend = time.time()
		print "foreground mask extracted in", tend - tstart, "seconds."


		# map foreground to lower dimension for faster clustering
		tstart = time.time()
		self.pca = RandomizedPCA(n_components=self.n_components).fit(mask_mat)
		print "Total explained variance:", sum(self.pca.explained_variance_ratio_)
		fore_pca = self.pca.transform(mask_mat)
		tend = time.time()
		print "edges mapped to a lower dimension in", tend - tstart, "seconds."

		# apply unsupervised clustering on each image.
		tstart = time.time()
		self.cluster = KMeans(n_clusters=self.n_clusters, n_jobs=self.n_jobs).fit(fore_pca)
		tend = time.time()
		print "edges clustered in", tend - tstart, "seconds."
		return self

	def predict(self, x):
		"""
		SHOULD RETURN IMAGE FOREGROUND TOO?
		:param x: lookup nearest cluster based on images used to train model (from fit)
		:return: clustering labels, one for each row of x
		"""
		if self.cluster is None or self.pca is None:
			print "fit method must be called first"
			return 1

		# use the background image given to extract foreground from x
		tstart = time.time()
		mask_mat = eigenback(self.back_vec, x, self.back_thres, self.fore_thres, rval='mask_mat', blur=True)
		# tend = time.time()
		# print "foreground mask extracted in", tend - tstart, "seconds."
		# TOO SLOW, TRY SOMETHING SIMPLER:
		# mask_mat = foreground_mask(self.back_vec, x, self.back_thres)
		tend = time.time()
		print "foreground mask extracted in", tend - tstart, "seconds."

		# map foreground to lower dimension using trained pca
		tstart = time.time()
		fore_pca = self.pca.transform(mask_mat)
		tend = time.time()
		print "edges mapped to a lower dimension in", tend - tstart, "seconds."

		# apply unsupervised clustering on each image using trained model
		tstart = time.time()
		cluster_labels = self.cluster.predict(fore_pca)
		tend = time.time()
		print "edges clustered in", tend - tstart, "seconds."
		return cluster_labels
