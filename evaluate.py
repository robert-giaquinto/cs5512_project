from __future__ import division
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn import preprocessing

# This is a script to do a "gut check" on the current results
# this will find accuracy using random forest
# for varying cluster sizes (window size doesn't matter much

def eval_random_forest(x_train_raw, y_train_raw, x_test_raw, y_test_raw, window_sz, num_clusters, n_jobs):
	train_dir = '/Users/robert/documents/umn/5512_AI2/project/data/train/'
	test_dir = '/Users/robert/documents/umn/5512_AI2/project/data/validation/'

	# cluster the low-dimensional data
	cluster = KMeans(n_clusters=num_clusters).fit(x_train_raw)
	cluster_train = cluster.predict(x_train_raw) + 1
	cluster_test = cluster.predict(x_test_raw) + 1

	# stack the results before using sliding window
	train_seq_ids = np.loadtxt(train_dir + "distorted_train_seq_ids.csv", delimiter=',', dtype=np.int16)
	train_frame_ids = np.loadtxt(train_dir + "distorted_train_frame_ids.csv", delimiter=',', dtype=np.int16)
	train_ids = np.loadtxt(train_dir + "distorted_train_ids.csv", delimiter=',', dtype=np.int16)
	train_frames = np.column_stack((y_train_raw, cluster_train, train_ids, train_seq_ids, train_frame_ids))

	test_seq_ids = np.loadtxt(test_dir + "distorted_test_seq_ids.csv", delimiter=',', dtype=np.int16)
	test_frame_ids = np.loadtxt(test_dir + "distorted_test_frame_ids.csv", delimiter=',', dtype=np.int16)
	test_ids = np.loadtxt(test_dir + "distorted_test_ids.csv", delimiter=',', dtype=np.int16)
	test_frames = np.column_stack((y_test_raw, cluster_test, test_ids, test_seq_ids, test_frame_ids))

	# use sliding window to put data in wide format
	train_window = sliding_window(train_frames, window_sz)
	test_window = sliding_window(test_frames, window_sz)

	# split into x and y
	y_train = train_window[:,0]
	y_test = test_window[:,0]
	x_train, x_test = convert_to_binary(train_window[:, 1:(window_sz+1)], test_window[:, 1:(window_sz+1)])


	# initialize classifier
	rf = RandomForestClassifier(n_estimators=250, n_jobs=n_jobs)
	rf = rf.fit(x_train, y_train)

	# return test set error
	return 1.0 - rf.score(x_test, y_test)


def convert_to_binary(x1, x2):
	num_features = x1.shape[1]
	x1_out = None
	x2_out = None

	for f in range(num_features):
		lb = preprocessing.LabelBinarizer()
		lb = lb.fit(x1[:,f])
		x1_bin = lb.transform(x1[:,f])
		x2_bin = lb.transform(x2[:,f])
		if f == 0:
			x1_out = x1_bin
			x2_out = x2_bin
		else:
			x1_out = np.column_stack((x1_out, x1_bin))
			x2_out = np.column_stack((x2_out, x2_bin))
	return x1_out, x2_out


# load the low-dimensional training and test data
train_dir = '/Users/robert/documents/umn/5512_AI2/project/data/train/'
test_dir = '/Users/robert/documents/umn/5512_AI2/project/data/validation/'
x_train_raw = np.loadtxt(train_dir + "polynomial_reduc_dim_training.csv", delimiter=',')
x_test_raw = np.loadtxt(test_dir + "polynomial_reduc_dim_testing.csv", delimiter=',')
# import labels
y_train_raw = np.loadtxt(train_dir + "polynomial_non_windowed_training.csv", delimiter=',')[:, 0]
y_test_raw = np.loadtxt(test_dir + "polynomial_non_windowed_testing.csv", delimiter=',')[:, 0]


cluster_range = [20, 25, 30, 35, 40, 45, 50]
test_error = []
for c in cluster_range:
	val = eval_random_forest(x_train_raw, y_train_raw, x_test_raw, y_test_raw, 10, c, n_jobs=1)
	print "C=" + str(c) + "\tError=" + str(round(val, 3))
	test_error.append(val)


# confusion matrix:
# pred = rf.predict(x_test)
# np.bincount(pred.astype(np.int16))
# confusion_matrix(y_test, pred)
#
# pred = rf.predict(x_train)
# np.bincount(pred.astype(np.int16))
# confusion_matrix(y_train, pred)







