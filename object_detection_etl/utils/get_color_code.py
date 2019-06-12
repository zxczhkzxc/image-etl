from sklearn.cluster import KMeans
from sklearn import metrics
import cv2
# By Adrian Rosebrock
import numpy as np
import cv2


# Load the image
# image = cv2.imread("red.png")
image = cv2.imread("blue.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Resize it
h, w, _ = image.shape
w_new = int(100 * w / max(w, h) )
h_new = int(100 * h / max(w, h) )

image = cv2.resize(image, (w_new, h_new));


# Reshape the image to be a list of pixels
image_array = image.reshape((image.shape[0] * image.shape[1], 3))
print image_array
# Clusters the pixels
clt = KMeans(n_clusters = 3)
clt.fit(image_array)

def centroid_histogram(clt):
        # grab the number of different clusters and create a histogram
        # based on the number of pixels assigned to each cluster
        numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
        (hist, _) = np.histogram(clt.labels_, bins = numLabels)

        # normalize the histogram, such that it sums to one
        hist = hist.astype("float")
        hist /= hist.sum()

        # return the histogram
        return hist


# Finds how many pixels are in each cluster
hist = centroid_histogram(clt)

# Sort the clusters according to how many pixel they have
zipped = zip (hist, clt.cluster_centers_)
zipped.sort(reverse=True, key=lambda x : x[0])
hist, clt.cluster_centers = zip(*zipped)

# By Adrian Rosebrock
import numpy as np
import cv2



bestSilhouette = -1
bestClusters = 0;

for clusters in range(2, 10):
	# Cluster colours
	clt = KMeans(n_clusters = clusters)
	clt.fit(image_array)

	# Validate clustering result
	silhouette = metrics.silhouette_score(image_array, clt.labels_, metric='euclidean')

	# Find the best one
	if silhouette > bestSilhouette:
		bestSilhouette = silhouette;
		bestClusters = clusters;

print(bestSilhouette)
print(bestClusters)


def plot_colors(hist, centroids):
	# initialize the bar chart representing the relative frequency
	# of each of the colors
	bar = np.zeros((50, 300, 3), dtype="uint8")
	startX = 0

	# loop over the percentage of each cluster and the color of
	# each cluster
	for (percent, color) in zip(hist, centroids):
		# plot the relative percentage of each cluster
		endX = startX + (percent * 300)
		cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
					  color.astype("uint8").tolist(), -1)
		startX = endX

	# return the bar chart
	return bar


bar = plot_colors(hist, clt.cluster_centers_)
import matplotlib.pyplot as plt
# show our color bart
plt.figure()
plt.axis("off")
plt.imshow(bar)
plt.show()
