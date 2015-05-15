from __future__ import division
import numpy as np
import random


# TODO: put this into a class similar to sklearn (with fit, predict methods)
# TODO: include restart capabilities in case kmeans "converges" after 1 iteration

def kmeans(data, num_clusters):
	max_iters = 150
	num_points = data.shape[0]
	num_features = data.shape[1]

	# initialize output arrays
	clusters = np.zeros([num_clusters, num_features])
	labels = np.zeros([num_points, 1])

	# intialize clusters from random points
	for c in range(num_clusters):
		# loop through each feature and select a random point
		random_point = np.zeros([1, num_features])
		for f in range(num_features):
			rand_int = random.randint(0, num_points)
			random_point[f] = data[rand_int, f]
		clusters[c, :] = random_point

	# threshold to stop early if algorithm converges
	threshold = 0.0001
	delta = float("inf")
	iter = 0
	while iter < max_iters and delta > threshold:
		iter += 1
		# first, assign each observation a cluster
		best_ssd = np.ones([num_points, 1]) * float("inf")
		for k in range(num_clusters):
			d = sum_squared_distance(data, clusters[k, :])
			for n in range(num_points):
				if d[n] < best_ssd[n]:
					labels[n] = k
					best_ssd[n] = d[n]

		# next, update locations of cluster centers
		old_clusters = clusters
		for k in range(num_clusters):
			clusters[k, :] = data[np.where(labels == k)[0], :].mean(axis=0)

		# how far did the centers move?
		delta = np.sum(np.abs(clusters - old_clusters))

		# how far did the means move
		delta = sum(sum(abs(clusters - old_clusters)))

	print "converged after", iter, "iterations"
	return clusters, labels


def sum_squared_distance(data, cluster):
	"""
	Assume data is normalized in order to give equal weight to each feature
	:param data:
	:param cluster:
	:return:
	"""
	ssd = np.linalg.norm(np.subtract(data, cluster), axis=1)
	return ssd


def assign_cluster(x, cluster_centers):
	"""
	Assign new data to a known cluster centroid
	:param x:
	:param cluster_centers:
	:return:
	"""
	num_points = x.shape[0]
	num_clusters = cluster_centers.shape[0]
	labels = np.zeros([num_points, 1])

	# loop through each cluster, keep note of closest cluster
	best_ssd = np.ones([num_points, 1]) * float("inf")
	for k in range(num_clusters):
		d = sum_squared_distance(x, cluster_centers[k, :])
		for n in range(num_points):
			if d[n] < best_ssd[n]:
				labels[n] = k
				best_ssd[n] = d[n]
	return labels

