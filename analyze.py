
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
