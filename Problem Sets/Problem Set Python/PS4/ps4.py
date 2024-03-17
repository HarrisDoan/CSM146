# -*- coding: utf-8 -*-
"""CS146_Winter2024_PS4.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Rk-uILdhujp5ragTxy6YQWsK7mGAKn5e

# PCA and k-means

## Setting up
"""

"""
Author      : Yi-Chieh Wu, Sriram Sankararaman
"""
import time

# numpy and scipy libraries
import numpy as np
from scipy import stats

# matplotlib libraries
import matplotlib.pyplot as plt
import collections

# To add your own Drive Run this cell.
from google.colab import drive
drive.mount('/content/drive')

import sys
# Change the path below to the path where your folder locates
# where you have util.py
### ========== TODO : START ========== ###
sys.path.append('/content/drive/My Drive/UCLA/Winter 2024/CSM146/Problem Set 4')
### ========== TODO : START ========== ###

import util
from util import *

"""## Point, Cluster and Set of Clusters classes"""

######################################################################
# classes
######################################################################

class Point(object) :

    def __init__(self, name, label, attrs) :
        """
        A data point.

        Attributes
        --------------------
            name  -- string, name
            label -- string, label
            attrs -- numpy arrray of shape (d, ) where d is the number of features
        """

        self.name = name
        self.label = label
        self.attrs = attrs


    #============================================================
    # utilities
    #============================================================

    def distance(self, other) :
        """
        Return Euclidean distance of this point with other point.

        Parameters
        --------------------
            other -- Point, point to which we are measuring distance

        Returns
        --------------------
            dist  -- float, Euclidean distance
        """
        # Euclidean distance metric
        return np.linalg.norm(self.attrs-other.attrs)


    def __str__(self) :
        """
        Return string representation.
        """
        return "%s : (%s, %s)" % (self.name, str(self.attrs), self.label)

class Cluster(object) :

    def __init__(self, points) :
        """
        A cluster (set of points).

        Attributes
        --------------------
            points -- list of Points, cluster elements
        """
        self.points = points


    def __str__(self) :
        """
        Return string representation.
        """
        s = ""
        for point in self.points :
            s += str(point)
        return s

    #============================================================
    # utilities
    #============================================================

    def purity(self) :
        """
        Compute cluster purity.

        Returns
        --------------------
            n           -- int, number of points in this cluster
            num_correct -- int, number of points in this cluster
                                with label equal to most common label in cluster
        """
        labels = []
        for p in self.points :
            labels.append(p.label)

        cluster_label, count = stats.mode(labels)
        return len(labels), np.float64(count)


    def centroid(self) :
        """
        Compute centroid of this cluster.

        Returns
        --------------------
            centroid -- Point, centroid of cluster
        """

        ### ========== TODO : START ========== ###
        # part 2b: implement
        # set the centroid label to any value (e.g. the most common label in this cluster)
        #attrs = [np.mean([p.attrs[i] for p in self.points]) for i in range(len(self.points[0].attrs))]
        all_attrs = np.array([p.attrs for p in self.points])
        attrs = np.mean(all_attrs, axis = 0)

        label = stats.mode([p.label for p in self.points])
        name = label
        centroid = Point(name, label, attrs)
        return centroid
        ### ========== TODO : END ========== ###

    def medoid(self) :
        """
        Compute medoid of this cluster, that is, the point in this cluster
        that is closest to all other points in this cluster.

        Returns
        --------------------
            medoid -- Point, medoid of this cluster
        """

        ### ========== TODO : START ========== ###
        # part 2b: implement
        pairwise_dist_matrix = []
        for p in self.points:
          temp_lst = []
          for p2 in self.points:
            temp_lst.append(p.distance(p2))
          pairwise_dist_matrix.append(temp_lst)
        row_sums = np.sum(pairwise_dist_matrix, axis=0).tolist()
        medoid = self.points[np.nanargmin(row_sums)]
        return medoid
        ### ========== TODO : END ========== ###


    def equivalent(self, other) :
        """
        Determine whether this cluster is equivalent to other cluster.
        Two clusters are equivalent if they contain the same set of points
        (not the same actual Point objects but the same geometric locations).

        Parameters
        --------------------
            other -- Cluster, cluster to which we are comparing this cluster

        Returns
        --------------------
            flag  -- bool, True if both clusters are equivalent or False otherwise
        """

        if len(self.points) != len(other.points) :
            return False

        matched = []
        for point1 in self.points :
            for point2 in other.points :
                if point1.distance(point2) == 0 and point2 not in matched :
                    matched.append(point2)
        return len(matched) == len(self.points)

class ClusterSet(object):

    def __init__(self) :
        """
        A cluster set (set of clusters).

        Parameters
        --------------------
            members -- list of Clusters, clusters that make up this set
        """
        self.members = []


    #============================================================
    # utilities
    #============================================================

    def centroids(self) :
        """
        Return centroids of each cluster in this cluster set.

        Returns
        --------------------
            centroids -- list of Points, centroids of each cluster in this cluster set
        """

        ### ========== TODO : START ========== ###
        # part 2b: implement
        centroids = [m.centroid() for m in self.members]
        return centroids
        ### ========== TODO : END ========== ###


    def medoids(self) :
        """
        Return medoids of each cluster in this cluster set.

        Returns
        --------------------
            medoids -- list of Points, medoids of each cluster in this cluster set
        """

        ### ========== TODO : START ========== ###
        # part 2b: implement
        medoids = [m.medoid() for m in self.members]
        return medoids
        ### ========== TODO : END ========== ###


    def score(self) :
        """
        Compute average purity across clusters in this cluster set.

        Returns
        --------------------
            score -- float, average purity
        """

        total_correct = 0
        total = 0
        for c in self.members :
            n, n_correct = c.purity()
            total += n
            total_correct += n_correct
        return total_correct / float(total)


    def equivalent(self, other) :
        """
        Determine whether this cluster set is equivalent to other cluster set.
        Two cluster sets are equivalent if they contain the same set of clusters
        (as computed by Cluster.equivalent(...)).

        Parameters
        --------------------
            other -- ClusterSet, cluster set to which we are comparing this cluster set

        Returns
        --------------------
            flag  -- bool, True if both cluster sets are equivalent or False otherwise
        """

        if len(self.members) != len(other.members):
            return False

        matched = []
        for cluster1 in self.members :
            for cluster2 in other.members :
                if cluster1.equivalent(cluster2) and cluster2 not in matched:
                    matched.append(cluster2)
        return len(matched) == len(self.members)


    #============================================================
    # manipulation
    #============================================================

    def add(self, cluster):
        """
        Add cluster to this cluster set (only if it does not already exist).

        If the cluster is already in this cluster set, raise a ValueError.

        Parameters
        --------------------
            cluster -- Cluster, cluster to add
        """

        if cluster in self.members :
            raise ValueError

        self.members.append(cluster)

"""## k-means and k-medoids algorithms"""

######################################################################
# k-means and k-medoids
######################################################################

def random_init(points, k) :
    """
    Randomly select k unique elements from points to be initial cluster centers.

    Parameters
    --------------------
        points         -- list of Points, dataset
        k              -- int, number of clusters

    Returns
    --------------------
        initial_points -- list of k Points, initial cluster centers
    """
    ### ========== TODO : START ========== ###
    # part 2c: implement (hint: use np.random.choice)
    return np.random.choice(points, size=k, replace=False).tolist()
    ### ========== TODO : END ========== ###


def cheat_init(points) :
    """
    Initialize clusters by cheating!

    Details
    - Let k be number of unique labels in dataset.
    - Group points into k clusters based on label (i.e. class) information.
    - Return medoid of each cluster as initial centers.

    Parameters
    --------------------
        points         -- list of Points, dataset

    Returns
    --------------------
        initial_points -- list of k Points, initial cluster centers
    """
    ### ========== TODO : START ========== ###
    # part 2f: implement
    initial_points = []
    labels = list(set([p.label for p in points]))
    for l in labels:
      c = Cluster([p for p in points if p.label == l])
      initial_points.append(c.medoid())

    return initial_points
    ### ========== TODO : END ========== ###


def kMeans(points, k, init='random', plot=False) :
    """
    Cluster points into k clusters using variations of k-means algorithm.

    Parameters
    --------------------
        points  -- list of Points, dataset
        k       -- int, number of clusters
        average -- method of ClusterSet
                   determines how to calculate average of points in cluster
                   allowable: ClusterSet.centroids, ClusterSet.medoids
        init    -- string, method of initialization
                   allowable:
                       'cheat'  -- use cheat_init to initialize clusters
                       'random' -- use random_init to initialize clusters
        plot    -- bool, True to plot clusters with corresponding averages
                         for each iteration of algorithm

    Returns
    --------------------
        k_clusters -- ClusterSet, k clusters
    """

    ### ========== TODO : START ========== ###
    # part 2c: implement
    # Hints:
    #   (1) On each iteration, keep track of the new cluster assignments
    #       in a separate data structure. Then use these assignments to create
    #       a new ClusterSet object and update the centroids.
    #   (2) Repeat until the clustering no longer changes.
    #   (3) To plot, use plot_clusters(...).
    #average = ClusterSet.centroids #or ClusterSet.medoids
    return kAverages(points, k, ClusterSet.centroids, init, plot)

def kAverages(points, k, average, init='random', plot=True):

    assignments = {}
    if init == 'cheat':
      initial_points = cheat_init(points)
    elif init == 'random':
      initial_points = random_init(points, k)

    for p in initial_points:
      assignments[p] = []


    for p in points:
      tmp_lst = [p.distance(c) for c in initial_points]
      closest_centroid = initial_points[np.nanargmin(tmp_lst)]
      assignments[closest_centroid].append(p)

    initial_clusterset = ClusterSet()
    for c in assignments.values():
      initial_clusterset.add(Cluster(c))

    itera = 0
    while True:

      if plot == True:
        plot_clusters(initial_clusterset, "Iteration " + str(itera), average)
      itera += 1

      tmp_assignments = {}
      #find new centroids/medoids
      tmp_initial_points = average(initial_clusterset)
      for p in tmp_initial_points:
        tmp_assignments[p] = []

      for p in points:
        tmp_lst = [p.distance(c) for c in tmp_initial_points]
        tmp_assignments[tmp_initial_points[np.nanargmin(tmp_lst)]].append(p)

      tmp_clusterset = ClusterSet()
      for c in tmp_assignments.values():
        tmp_clusterset.add(Cluster(c))

      if tmp_clusterset.equivalent(initial_clusterset):
        break
      else:
        initial_clusterset = tmp_clusterset
    if average == ClusterSet.centroids:
      print("kMeans: " + str(itera))
    else:
      print("kMedoids: " + str(itera))
    k_clusters = initial_clusterset

    #k_clusters = ClusterSet()
    return k_clusters
    ### ========== TODO : END ========== ###

def kMedoids(points, k, init='random', plot=False) :
    """
    Cluster points in k clusters using k-medoids clustering.
    See kMeans(...).
    """
    ### ========== TODO : START ========== ###
    # part 2e: implement

    #k_clusters = ClusterSet()
    #return k_clusters
    return kAverages(points, k, ClusterSet.medoids, init, plot)

    ### ========== TODO : END ========== ###

"""## Utilities"""

######################################################################
# helper functions
######################################################################

def build_face_image_points(X, y) :
    """
    Translate images to (labeled) points.

    Parameters
    --------------------
        X     -- numpy array of shape (n,d), features (each row is one image)
        y     -- numpy array of shape (n,), targets

    Returns
    --------------------
        point -- list of Points, dataset (one point for each image)
    """

    n,d = X.shape

    images = collections.defaultdict(list) # key = class, val = list of images with this class
    for i in range(n) :
        images[y[i]].append(X[i,:])

    points = []
    for face in images :
        count = 0
        for im in images[face] :
            points.append(Point(str(face) + '_' + str(count), face, im))
            count += 1

    return points


def plot_clusters(clusters, title, average) :
    """
    Plot clusters along with average points of each cluster.

    Parameters
    --------------------
        clusters -- ClusterSet, clusters to plot
        title    -- string, plot title
        average  -- method of ClusterSet
                    determines how to calculate average of points in cluster
                    allowable: ClusterSet.centroids, ClusterSet.medoids
    """

    plt.figure()
    np.random.seed(20)
    label = 0
    colors = {}
    centroids = average(clusters)
    for c in centroids :
        coord = c.attrs
        plt.plot(coord[0],coord[1], 'ok', markersize=12)
    for cluster in clusters.members :
        label += 1
        colors[label] = np.random.rand(3,)
        for point in cluster.points :
            coord = point.attrs
            plt.plot(coord[0], coord[1], 'o', color=colors[label])
    plt.title(title)
    plt.show()


def generate_points_2d(N, seed=1234) :
    """
    Generate toy dataset of 3 clusters each with N points.

    Parameters
    --------------------
        N      -- int, number of points to generate per cluster
        seed   -- random seed

    Returns
    --------------------
        points -- list of Points, dataset
    """
    np.random.seed(seed)

    mu = [[0,0.5], [1,1], [2,0.5]]
    sigma = [[0.1,0.1], [0.25,0.25], [0.15,0.15]]

    label = 0
    points = []
    for m,s in zip(mu, sigma) :
        label += 1
        for i in range(N) :
            x = random_sample_2d(m, s)
            points.append(Point(str(label)+'_'+str(i), label, x))

    return points

"""## Main function"""

######################################################################
# main
######################################################################

# def main() :
#     ### ========== TODO : START ========== ###
#     # part 1: explore LFW data set
#     X, y = get_lfw_data()
#     for img in X[:5]:
#       show_image(img)
#     mean_imgs = np.mean(X, axis = 0)
#     show_image(mean_imgs)


#     #explore the effect of using more or fewer dimensions to represent images
#     #project the original data into the lower-dimensional space
#     #Z, Ul = apply_PCA_from_Eig(X, U, 1, mu)
#     #recontruct high-dimensional images out of lower-dimensional ones
#     #X_rec = reconstruct_from_PCA(Z, Ul, mu)
#     #U2, mu2 = util.PCA(X_rec)
#     #print(U[:,0])
#     #print([U2[k][:,0] for k in range(U2.shape[0])])
#     #show_image(vec_to_image([U2[k][:,0] for k in range(U2.shape[0])]))
#     #plot gallery
#     #plot_gallery([vec_to_image(X_rec[:,i]) for i in range(12)])
#     U, mu = PCA(X)
#     plot_gallery([vec_to_image(U[:,i]) for i in range(12)])


#     lst = [1, 10, 50, 100, 500, 1288]
#     for l in lst:
#         Z, Ul = apply_PCA_from_Eig(X, U, l, mu)
#         X_rec = reconstruct_from_PCA(Z, Ul, mu)
#         print("With l =", l)
#         plot_gallery(X_rec)
#     ### ========== TODO : END ========== ###



#     ### ========== TODO : START ========== ###
#     # part 2d-2f: cluster toy dataset
#     # np.random.seed(1234) ## don't change the seed !!
#     # points = generate_points_2d(50)
#     # kMeans(points, 3, init='random', plot=True)
#     # kMedoids(points, 3, init='random', plot=True)

#     # print("Testing with cheat_init...")
#     # kMeans(points, 3, init='cheat', plot=True)
#     # kMedoids(points, 3, init='cheat', plot=True)

#     np.random.seed(1234)
#     k = 3
#     for i in range(1):
#         points = generate_points_2d(20)
#         k_clusters = kMeans(points, k, init="cheat", plot=True)
#         k_clusters = kMedoids(points, k, init="cheat", plot=True)
#     ### ========== TODO : END ========== ###



#     ### ========== TODO : START ========== ###
#     # part 3a: cluster faces
#     np.random.seed(1234)  # don't change the seed !!

#     X1, y1 = util.limit_pics(X, y, [4, 6, 13, 16], 40)
#     points = build_face_image_points(X1, y1)

#     results_kMeans = []
#     times_kMeans = []
#     results_kMedoids = []
#     times_kMedoids = []

#     for i in range(10):
#         # kMeans
#         start_time = time.time()
#         k_clusters = kMeans(points, 4, init='random', plot=False)
#         end_time = time.time()
#         results_kMeans.append(k_clusters.score())
#         times_kMeans.append(end_time - start_time)

#         # kMedoids
#         start_time = time.time()
#         k_clusters = kMedoids(points, 4, init='random', plot=False)
#         end_time = time.time()
#         results_kMedoids.append(k_clusters.score())
#         times_kMedoids.append(end_time - start_time)

#     # Computing and printing kMeans statistics
#     print("(kMeans) Average Purity: {:.6f}, Min: {:.6f}, Max: {:.6f}".format(np.mean(results_kMeans), np.min(results_kMeans), np.max(results_kMeans)))
#     print("(kMeans) Average Time: {:.6f} seconds, Min: {:.6f} seconds, Max: {:.6f} seconds".format(np.mean(times_kMeans), np.min(times_kMeans), np.max(times_kMeans)))

#     # Computing and printing kMedoids statistics
#     print("(kMedoids) Average Purity: {:.6f}, Min: {:.6f}, Max: {:.6f}".format(np.mean(results_kMedoids), np.min(results_kMedoids), np.max(results_kMedoids)))
#     print("(kMedoids) Average Time: {:.6f} seconds, Min: {:.6f} seconds, Max: {:.6f} seconds".format(np.mean(times_kMedoids), np.min(times_kMedoids), np.max(times_kMedoids)))
#     ### ========== TODO : END ========== ###



#     # part 3b: explore effect of lower-dimensional representations on clustering performance
#     np.random.seed(1234) ## don't change the seed !!

#     Xb, yb = util.limit_pics(X, y, [4, 13], 40)
#     #compute the principal components for the entire image dataset
#     Ub, mub = util.PCA(X)

#     num_components = [k for k in range(1, 50, 2)]
#     results_kMeansb = []
#     results_kMedoidsb = []

#     for i in range(1, 50, 2):
#       #project the newly generated dataset into a lower dimension
#       Z, Ul = apply_PCA_from_Eig(Xb, Ub, i, mub)
#       X_recb = reconstruct_from_PCA(Z, Ul, mub)
#       #generate new list of points using build_face_image_points
#       pointsb = build_face_image_points(X_recb, yb)

#       #compute the scores of each clustering algorithm
#       cs1 = kMeans(pointsb, 2, init='cheat', plot=False)
#       cs2 = kMedoids(pointsb, 2, init='cheat', plot=False)
#       results_kMeansb.append(cs1.score())
#       results_kMedoidsb.append(cs2.score())

#     plt.figure(figsize=(12.8, 9.6))
#     plt.plot(num_components, results_kMeansb, label="kMeans", color="yellow", linestyle='-', marker='o')
#     plt.plot(num_components, results_kMedoidsb, label="kMedoids", color="blue", linestyle='-', marker='o')
#     plt.xlabel('number of components')
#     plt.ylabel('score')
#     plt.grid(True)
#     plt.legend()
#     plt.show()





#     # part 3c: determine ``most discriminative'' and ``least discriminative'' pairs of images
#     np.random.seed(1234) ## don't change the seed !!
#     #19 different people
#     perms = [(i,j) for i in range(19) for j in range(i,19) if i!=j]
#     dict_scores = {"min":[np.Infinity,0,0], "max":[0,0,0]}
#     for p in perms:
#       Xc, yc = util.limit_pics(X, y, p, 40)
#       pointsc = build_face_image_points(Xc, yc)
#       cs = kMedoids(pointsc, 2, init='cheat', plot=False)
#       cscore = cs.score()
#       if cscore > dict_scores["max"][0]:
#         dict_scores["max"] = [cscore, p[0], p[1]]
#       if cscore < dict_scores["min"][0]:
#         dict_scores["min"] = [cscore, p[0], p[1]]

#     print(dict_scores)
#     plot_representative_images(X, y, [dict_scores["max"][1], dict_scores["max"][2]], title='easy to discriminate')
#     plot_representative_images(X, y, [dict_scores["min"][1], dict_scores["min"][2]], title='difficult to discriminate')
#     ### ========== TODO : END ========== ###


# if __name__ == "__main__" :
#     main()

def main() :
    ### ========== TODO : START ========== ###
    # part 1: explore LFW data set
    #get LFW dataset with labels
    X, y = get_lfw_data()
    #plot a couple of input images
    show_image(X[0])
    show_image(X[1])
    show_image(X[2])
    show_image(X[3])
    #compute the mean of all the images and plot it
    img_mean = np.mean(X, 0)
    show_image(img_mean)

    #perform PCA on the data
    U, mu = util.PCA(X)
    #columns of U are principal components, mu is mean of data
    plot_gallery([vec_to_image(U[:,i]) for i in range(12)])



    #explore the effect of using more or fewer dimensions to represent images
    #project the original data into the lower-dimensional space
    #Z, Ul = apply_PCA_from_Eig(X, U, 1, mu)
    #recontruct high-dimensional images out of lower-dimensional ones
    #X_rec = reconstruct_from_PCA(Z, Ul, mu)
    #U2, mu2 = util.PCA(X_rec)
    #print(U[:,0])
    #print([U2[k][:,0] for k in range(U2.shape[0])])
    #show_image(vec_to_image([U2[k][:,0] for k in range(U2.shape[0])]))
    #plot gallery
    #plot_gallery([vec_to_image(X_rec[:,i]) for i in range(12)])


    lst = [1, 10, 50, 100, 500, 1288]
    for l in lst:
        Z, Ul = apply_PCA_from_Eig(X, U, l, mu)
        X_rec = reconstruct_from_PCA(Z, Ul, mu)
        plot_gallery([vec_to_image(X_rec[i]) for i in range(12)], title='l=' + str(l))
    ### ========== TODO : END ========== ###



    ### ========== TODO : START ========== ###
    # part 2d-2f: cluster toy dataset
    np.random.seed(1234)
    points = generate_points_2d(20)
    kMeans(points, 3, init='random', plot=True)
    kMedoids(points, 3, init='random', plot=True)

    print("Testing with cheat_init...")
    kMeans(points, 3, init='cheat', plot=True)
    kMedoids(points, 3, init='cheat', plot=True)
    ### ========== TODO : END ========== ###



    ### ========== TODO : START ========== ###
    # part 3a: cluster faces
    np.random.seed(1234)


    X1, y1 = util.limit_pics(X, y, [4, 6, 13, 16], 40)
    points = build_face_image_points(X1, y1)

    # Initialize lists to store scores and times
    results_kMeans = []
    times_kMeans = []
    results_kMedoids = []
    times_kMedoids = []

    # Perform clustering and time each call
    for i in range(10):
        start = time.time()
        cs1 = kMeans(points, 4, init='random', plot=False)
        end = time.time()
        results_kMeans.append(cs1.score())
        times_kMeans.append(end - start)

        start = time.time()
        cs2 = kMedoids(points, 4, init='random', plot=False)
        end = time.time()
        results_kMedoids.append(cs2.score())
        times_kMedoids.append(end - start)

    # Calculate and print the average, minimum, and maximum scores
    print("(kMeans) Score - average: %f, min: %f, max: %f" % (np.mean(results_kMeans), np.min(results_kMeans), np.max(results_kMeans)))
    print("(kMedoids) Score - average: %f, min: %f, max: %f" % (np.mean(results_kMedoids), np.min(results_kMedoids), np.max(results_kMedoids)))

    # Calculate and print the average, minimum, and maximum times
    print("(kMeans) Time - average: %f, min: %f, max: %f seconds" % (np.mean(times_kMeans), np.min(times_kMeans), np.max(times_kMeans)))
    print("(kMedoids) Time - average: %f, min: %f, max: %f seconds" % (np.mean(times_kMedoids), np.min(times_kMedoids), np.max(times_kMedoids)))



    # part 3b: explore effect of lower-dimensional representations on clustering performance
    np.random.seed(1234)
    Xb, yb = util.limit_pics(X, y, [4, 13], 40)
    #compute the principal components for the entire image dataset
    Ub, mub = util.PCA(X)

    num_components = [k for k in range(1, 50, 2)]
    results_kMeansb = []
    results_kMedoidsb = []

    for i in range(1, 50, 2):
      #project the newly generated dataset into a lower dimension
      Z, Ul = apply_PCA_from_Eig(Xb, Ub, i, mub)
      X_recb = reconstruct_from_PCA(Z, Ul, mub)
      #generate new list of points using build_face_image_points
      pointsb = build_face_image_points(X_recb, yb)

      #compute the scores of each clustering algorithm
      cs1 = kMeans(pointsb, 2, init='cheat', plot=False)
      cs2 = kMedoids(pointsb, 2, init='cheat', plot=False)
      results_kMeansb.append(cs1.score())
      results_kMedoidsb.append(cs2.score())

    plt.figure(figsize=(12.8,9.6))
    plt.plot(num_components, results_kMeansb, color = 'yellow', label="kMeans")
    plt.plot(num_components, results_kMedoidsb, color = 'blue', label="kMedoids")
    plt.title('kMeans/kMedoids Scores vs PCA Amount')
    plt.xlabel('Number of Components')
    plt.ylabel('Purity Score')
    plt.grid(True)
    plt.legend()
    plt.show




    # part 3c: determine ``most discriminative'' and ``least discriminative'' pairs of images
    np.random.seed(1234)
    #19 different people
    perms = [(i,j) for i in range(19) for j in range(i,19) if i!=j]
    dict_scores = {"min":[np.Infinity,0,0], "max":[0,0,0]}
    for p in perms:
      Xc, yc = util.limit_pics(X, y, p, 40)
      pointsc = build_face_image_points(Xc, yc)
      cs = kMedoids(pointsc, 2, init='cheat', plot=False)
      cscore = cs.score()
      if cscore > dict_scores["max"][0]:
        dict_scores["max"] = [cscore, p[0], p[1]]
      if cscore < dict_scores["min"][0]:
        dict_scores["min"] = [cscore, p[0], p[1]]

    print(dict_scores)
    plot_representative_images(X, y, [dict_scores["max"][1], dict_scores["max"][2]], title='easy to discriminate')
    plot_representative_images(X, y, [dict_scores["min"][1], dict_scores["min"][2]], title='difficult to discriminate')
    ### ========== TODO : END ========== ###


if __name__ == "__main__" :
    main()

