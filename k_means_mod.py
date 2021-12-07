#!/usr/bin/env python
import numpy as np


def euc_dist(x, y):
    """ Calculate Euclidean distance between two points """
    dist = np.sqrt(((x - y) ** 2).sum())

    return dist


class KmeansMod(object):
    def __init__(self, data=None, k=3, init="rand", max_iter=100, threshold=0.0001):
        """
        :param data: 2D data array
        :param k: Number of desired clusters
        :param init:    "rand": Randomly selecting k data points as centroids
                        "farthest": Select a centroid then find the farthest
                                    point as the next centroid, and so forth
                        "seg":  Slice the data space to k equal segments then find the centroid
                        "sort": Sort the data based on the distance to a specific starting point
                                then select k centroids at a m/k step
        :param max_iter: Maximum allowed iterations
        :param threshold: Convergence tolerance
        """
        self.data = data
        self.k = k
        init_methods = {"rand": self.rand_init,
                        "farthest": self.farthest_init,
                        "seg": self.seg_init,
                        "sort": self.sort_init,
                        }
        self.init = init_methods[init]
        self.max_iter = max_iter
        self.threshold = threshold

    def rand_init(self):
        """ Initiate k centroids by randomly selecting k data points (straightforward). """
        random_idx = [np.random.randint(len(self.data)) for i in range(self.k)]  # Random index list, size (k)
        centroids = self.data[random_idx]  # Centroid list, size (k,m)

        return centroids

    def farthest_init(self):
        """ Maximize the distances between initial centroids
            Performance may not be desirable when the dataset has outliers.
        """
        m = self.data.shape[0]
        n = self.data.shape[1]

        centroids = np.empty([self.k, n])  # Initialize the centroid array
        indexes = []  # Use List for indexing

        for i in range(self.k):
            if i == 0:  # Randomly select a point as the first centroid
                idx = np.random.randint(m)
                centroids[i] = self.data[idx]
                indexes.append(idx)
            else:
                dist = np.empty([m])
                for num, row in enumerate(self.data):
                    # Get the distance between previous centroid and data
                    dist[num] = euc_dist(row, centroids[i - 1])
                dist[indexes] = -1  # Points in the centroid list won't be selected again
                idx = np.argmax(dist)  # Find the row index with the longest distance
                centroids[i] = self.data[idx]
                indexes.append(idx)  # Save the index to the list for next iteration

        return centroids

    def seg_init(self):
        """ Slice the data space to k segments then calculate initial centroids.
             *** Deterministic ***
        """
        m = self.data.shape[0]
        n = self.data.shape[1]

        # Initialize centroids array
        centroids = np.empty([self.k, n])

        # Sum the features by rows and add the sums as a column to the original dataset
        sum_col = self.data.sum(axis=1)
        data_comp = np.asarray(np.append(np.mat(sum_col).T, self.data, axis=1))
        data_comp = data_comp[data_comp[:, 0].argsort()]  # Sort the composite dataset

        # Segment size and convert to integer for indexing
        seg_size = int(np.floor(m / self.k))

        # Evenly slice the dataset to k segments then calculate means
        for i in range(self.k):
            if i < self.k - 1:
                centroids[i:] = data_comp[i * seg_size:(i + 1) * seg_size, 1:].mean(axis=0)
            else:
                centroids[i:] = data_comp[i * seg_size:, 1:].mean(axis=0)

        return centroids

    def sort_init(self):
        """ Sort the data points based on the distance to a randomly selected point,
            then select every m/k (e.g., 50 when m = 150 and k = 3) point in the
            sorted dataset as the centroids.
        """
        m = self.data.shape[0]

        idx = np.random.randint(m)
        init_point = self.data[idx]  # Select a random start point

        # Calculate the distance to the start point by rows and
        # add the sums as a column to the original dataset
        dist = np.empty([m])
        for num, row in enumerate(self.data):
            dist[num] = euc_dist(row, init_point)

        data_comp = np.asarray(np.append(np.mat(dist).T, self.data, axis=1))
        data_comp = data_comp[data_comp[:,0].argsort()]

        seg_size = int(np.floor(m / self.k))

        indexes = []
        for i in range(self.k):
            indexes.append((i + 1) * seg_size - 1)

        centroids = np.asarray(data_comp[indexes, 1:])  # Convert the centroid matrix slice to array

        return centroids

    def get_labels(self, centroids):
        """ Predict labels based on given dataset and centroids
            Initialize an array for saving k distances between the data points and the centroids.
        """
        dist_mat = np.empty([len(self.data), self.k])

        for n, centroid in enumerate(centroids):
            # Replace the nth column with the distance array to centroids[n]
            dist_mat[:, n] = np.array([euc_dist(self.data[row], centroid) for row in range(len(self.data))])

        # Acquire an array of the col_index of the smallest distance as the labels
        labels = np.argmin(dist_mat, axis=1)

        # If one of the centroid is not shown in the label, re-initialize the whole process
        if len(np.unique(labels)) != len(centroids):
            centroids = self.init()
            return self.get_labels(centroids)
        else:
            return labels

    def kmeans(self):

        prev_centroids = self.init()
        prev_labels = self.get_labels(prev_centroids)

        # Initialize the centroids array
        centroids = np.empty([self.k, len(self.data[0])])
        iter_count = 0

        while True:
            if iter_count > self.max_iter:  # Break the loop if the iteration count exceeds max_iter
                break
            for i in range(self.k):
                # Get new centroid by calculating the mean value of the features in the same cluster
                centroid = self.data[prev_labels == i].mean(axis=0)
                centroids[i] = centroid

            print(f"Iteration {iter_count} centroids:\n", centroids,
                  "\nPrevious iteration centroids:\n", prev_centroids)  # For debug

            # Get new labels
            labels = self.get_labels(centroids)

            # Check the if the difference is smaller than the tolerance (threshold)
            diff = centroids - prev_centroids
            if (labels == prev_labels).all() or (diff < self.threshold).all():
                break

            # Assign the current centroids to prev_centroids for the next iteration
            prev_centroids = centroids
            prev_labels = labels

            iter_count += 1  # Iteration counter +1

        return centroids, labels
