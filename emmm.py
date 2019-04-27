# emmm.py: EM algorithm on discrete random points.
import argparse
from math import exp, inf, log, pi

import matplotlib.pyplot as plt
import numpy as np

from kmeans import randCent

X = None
mju = None
sigma = None
pre = None
gamma = None
n = 0
k = 0


def multiNormal(x, mean, stdvar):
    dis = x - mean
    frac = -0.5 * np.dot(dis, np.dot(np.linalg.inv(stdvar), dis))
    return exp(frac) / ((2 * pi) ** (len(x) / 2) * np.linalg.det(stdvar) ** 0.5)


def likelihood():
    like = 0
    for i in range(n):
        w = 0
        for j in range(k):
            w += pre[j] * multiNormal(X[i], mju[j], sigma[j])
        like += log(w)
    return like


def estep():
    slot = np.zeros(k)
    for i in range(n):
        denom = 0
        for j in range(k):
            slot[j] = pre[j] * multiNormal(X[i], mju[j], sigma[j])
            denom += slot[j]
        for t in range(k):
            gamma[i, t] = slot[t]/denom


def mstep():
    dim = X.shape[1]
    for i in range(k):
        Nk = np.sum(gamma[:, i])
        mju[i] = np.dot(gamma[:, i], X) / Nk
        dis = X - mju[i]  # row vector matrix
        #sigma[i] = np.dot(gamma[:, i] * dis.T, dis) / Nk
        tsig = np.zeros((dim, dim))
        for j in range(n):
            jcol = dis[j].reshape((dim, 1)) 
            tsig += gamma[j, i] * np.dot(jcol, jcol.T)
        sigma[i] = tsig / Nk
        pre[i] = Nk/n


def main():
    global X, n, k, pre, mju, sigma, gamma
    data = np.loadtxt('rawdata.csv', delimiter=',')
    parser = argparse.ArgumentParser(
        description='EM clustering program.')
    parser.add_argument('-k', type=int, default=3,
                        help='The number of clusters. Default is 3.')
    args = parser.parse_args()
    X = data[:, 0:-1]
    n = X.shape[0]
    k = args.k
    pre = [1/k for i in range(k)]
    # Generate random mju initial values
    mju = randCent(X, k)
    sigma = [np.eye(X.shape[1]) for i in range(k)]
    gamma = np.zeros((n, k))
    step, like = EM()
    print('Total steps:', step)
    print('Likelihood:', like)
    print('Centers: \n', mju)
    print('Constituents: \n', gamma)
    # Drawing
    colors=['black', 'blue', 'green', 'lime', 'maroon', 'olive', 'orange', 'purple', 'red', 'teal', 'yellow']
    np.random.shuffle(colors)
    major = np.argmax(gamma,axis=1)
    for i in range(k):
        plt.scatter(mju[i, 0], mju[i, 1], c=colors[i], marker='*')
        cluPoints = X[np.where(major == i)]
        plt.scatter(cluPoints[:, 0], cluPoints[:, 1], c=colors[i])
    plt.show()


def EM():
    step = 0
    like = 0
    last_like = inf
    while True:
        estep()
        mstep()
        like = likelihood()
        step += 1
        print(f'Likelihood after {step} steps = {like}')
        if abs(like - last_like) <= 1e-6:
            return step, like
        else:
            last_like = like


if __name__ == '__main__':
    main()
