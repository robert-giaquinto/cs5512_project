from __future__ import division
from sklearn.decomposition import RandomizedPCA, KernelPCA
from sklearn.lda import LDA
import time
from kmeans import kmeans, assign_cluster
from sklearn.cluster import KMeans

# NOTE: USING SKLEARN'S KMEANS FOR PRODUCTION (IT'S FASTER, SAFER)
# BUT OUR IMPLEMENTATION WORKS FINE
# (most of the time, it doesn't have restart capability built in, yet)


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
	def __init__(self, n_clusters=30, n_components=50, method=None, n_jobs=1, use_sklearn=True):

		# parameters (optional)
		self.n_clusters = n_clusters
		self.n_components = n_components
		self.method = method
		self.n_jobs=n_jobs
		self.use_sklearn=use_sklearn

		# to be assigned later
		self.cluster = None
		self.dim_reduc = None

	def fit(self, x, y=None):
		"""
		:param x: a matrix of images to cluster on
		"""
		# map foreground to lower dimension for faster clustering
		if self.method == 'RandomizedPCA':
			self.dim_reduc = RandomizedPCA(n_components=self.n_components)
		elif self.method == "KernelPCA":
			self.dim_reduc = KernelPCA(n_components=self.n_components, kernel="poly")
		elif self.method == "LDA":
			self.dim_reduc = LDA(n_components=self.n_components)
		else:
			print "valid method name not given, defaulting to randomized pca"
			self.dim_reduc = RandomizedPCA(n_components=self.n_components)

		# fit dimension reduction
		start_time = time.time()
		if self.method == "LDA":
			if y is not None:
				self.dim_reduc = self.dim_reduc.fit(x, y)
			else:
				print "must provide a y to use LDA"
				return 1
		else:
			self.dim_reduc = self.dim_reduc.fit(x)
		end_time = time.time()
		print "time to fit dimension reduction:", round(end_time - start_time, 1), "seconds"

		# apply dimension reduction to data
		start_time = time.time()
		x_reduc = self.dim_reduc.transform(x)
		end_time = time.time()
		print "time to apply dimension reduction:", round(end_time - start_time, 1), "seconds"

		# apply unsupervised clustering on each image.
		start_time = time.time()
		if self.use_sklearn:
			self.cluster = KMeans(n_clusters=self.n_clusters, n_jobs=self.n_jobs).fit(x_reduc)
		else:
			self.cluster, labels = kmeans(x_reduc, self.n_clusters)
		end_time = time.time()
		print "time to fit clustering:", round(end_time - start_time, 1), "seconds"
		return self

	def predict(self, x):
		"""
		:param x: lookup nearest cluster based on images used to train model (from fit)
		:return: clustering labels, one for each row of x
		"""
		if self.cluster is None or self.dim_reduc is None:
			print "fit method must be called first"
			return 1

		# map foreground to lower dimension using trained pca
		x_reduc = self.dim_reduc.transform(x)
		print "dimensions reduced"

		# apply unsupervised clustering on each image using trained model
		# set cluster labels to start at 1
		if self.use_sklearn:
			cluster_labels = self.cluster.predict(x_reduc) + 1
		else:
			cluster_labels = assign_cluster(x_reduc, self.cluster)
		print "clusters assigned"
		return cluster_labels
