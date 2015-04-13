from __future__ import division
from sklearn.cluster import KMeans
from sklearn.decomposition import RandomizedPCA, IncrementalPCA, KernelPCA
# from sklearn.qda import QDA
from sklearn.lda import LDA


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
	def __init__(self, n_clusters=30, n_components=50, method=None, n_jobs=1,):

		# parameters (optional)
		self.n_clusters = n_clusters
		self.n_components = n_components
		self.n_jobs = n_jobs
		self.method = method

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
		elif self.method == 'IncrementalPCA':
			self.dim_reduc = IncrementalPCA(n_components=self.n_components)
		elif self.method == "KernelPCA":
			self.dim_reduc = KernelPCA(n_components=self.n_components, kernel="rbf", fit_inverse_transform=True)
		elif self.method == "LDA":
			self.dim_reduc = LDA(n_components=self.n_components)
		else:
			print "valid method name not given, defaulting to randomized pca"
			self.dim_reduc = RandomizedPCA(n_components=self.n_components)

		# fit dimension reduction
		if self.method == "LDA":
			if y is not None:
				self.dim_reduc = self.dim_reduc.fit(x, y)
			else:
				print "must provide a y to use LDA"
				return 1
		else:
			self.dim_reduc = self.dim_reduc.fit(x)

		# apply dimension reduction to data
		x_reduc = self.dim_reduc.transform(x)
		del x

		# apply unsupervised clustering on each image.
		self.cluster = KMeans(n_clusters=self.n_clusters, n_jobs=self.n_jobs).fit(x_reduc)
		return self

	def predict(self, x, y=None):
		"""
		:param x: lookup nearest cluster based on images used to train model (from fit)
		:return: clustering labels, one for each row of x
		"""
		if self.cluster is None or self.dim_reduc is None:
			print "fit method must be called first"
			return 1

		# map foreground to lower dimension using trained pca
		x_reduc = self.dim_reduc.transform(x)

		# apply unsupervised clustering on each image using trained model
		cluster_labels = self.cluster.predict(x_reduc)
		return cluster_labels
