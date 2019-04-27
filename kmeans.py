import argparse
from math import inf

import matplotlib.pyplot as plt
import numpy as np


# Create k random initial center points
def randCent(dataSet, k):
    n = np.shape(dataSet)[1]
    cpoints = np.zeros((k, n))
    # create centroid mat
    for j in range(n):
        # create random cluster centers, within bounds of each dimension
        minJ = np.min(dataSet[:, j])
        rangeJ = float(np.max(dataSet[:, j]) - minJ)
        cpoints[:, j] = minJ + rangeJ * np.random.rand(k)
    return cpoints


# Euclidean distance
def eucDist(pA, pB):
    return np.linalg.norm(pA - pB)


# K-means main method.
def kmeans(dataSet, k, distMeasure=eucDist, centerGen=randCent):
    m = np.shape(dataSet)[0]  # number of data
    cluResult = np.zeros((m, 2))  # center and distance of each point
    centers = centerGen(dataSet, k)  # the k centers
    cluChanged = True
    while cluChanged:
        cluChanged = False
        for i in range(m):
            minDist = inf
            minIndex = -1
            for j in range(k):
                # Step 1: find the nearest center for each point
                newDist = distMeasure(centers[j, :], dataSet[i, :])
                if newDist < minDist:
                    minDist = newDist
                    minIndex = j
            if cluResult[i, 0] != minIndex:
                cluChanged = True
                cluResult[i, :] = minIndex, minDist
        # Step 2: re-evaluate the center points
        for cent in range(k):
            ptsInClust = dataSet[np.where(cluResult[:, 0] == cent)]
            centers[cent, :] = np.mean(ptsInClust, axis=0)
    return centers, cluResult


def main():
    data = np.loadtxt('rawdata.csv', delimiter=',')
    parser = argparse.ArgumentParser(
        description='K-means clustering program.')
    parser.add_argument('-k', type=int, default=3,
                        help='The number of clusters. Default is 3.')
    args = parser.parse_args()
    dataSet = data[:, 0:-1]
    centers, cluResult = kmeans(dataSet, args.k)
    print('Centers:', centers)
    print('Clustering result:', cluResult)
    # Drawing
    colors=['black', 'blue', 'green', 'lime', 'maroon', 'olive', 'orange', 'purple', 'red', 'teal', 'yellow']
    np.random.shuffle(colors)
    for i in range(len(centers)):
        center=centers[i]
        cluPoint=dataSet[np.where(cluResult[:,0]==i)]
        plt.scatter(center[0],center[1],c=colors[i],marker='*')
        plt.scatter(cluPoint[:,0],cluPoint[:,1], c=colors[i])
    plt.show()


if __name__ == '__main__':
    main()
