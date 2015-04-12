from __future__ import division
from sklearn.cluster import KMeans
from sklearn.decomposition import RandomizedPCA


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
	def __init__(self, n_clusters=30, n_components=50, n_jobs=1):

		# parameters (optional)
		self.n_clusters = n_clusters
		self.n_components = n_components
		self.n_jobs = n_jobs

		# to be assigned later
		self.cluster = None
		self.pca = None

	def fit(self, x):
		"""
		:param x: a matrix of images to cluster on
		"""
		# map foreground to lower dimension for faster clustering
		self.pca = RandomizedPCA(n_components=self.n_components).fit(x)
		x_pca = self.pca.transform(x)

		# apply unsupervised clustering on each image.
		self.cluster = KMeans(n_clusters=self.n_clusters, n_jobs=self.n_jobs).fit(x_pca)
		return self

	def predict(self, x):
		"""
		:param x: lookup nearest cluster based on images used to train model (from fit)
		:return: clustering labels, one for each row of x
		"""
		if self.cluster is None or self.pca is None:
			print "fit method must be called first"
			return 1

		# map foreground to lower dimension using trained pca
		x_pca = self.pca.transform(x)

		# apply unsupervised clustering on each image using trained model
		cluster_labels = self.cluster.predict(x_pca)
		return cluster_labels
