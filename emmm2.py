# emmm2.py: EM algorithm on wine dataset.
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
    frac = -0.5 * np.dot(np.dot(dis, np.linalg.inv(stdvar)), dis)
    return np.exp(frac) / ((2 * pi) ** (len(x) / 2) * np.linalg.det(stdvar) ** 0.5)


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
    for i in range(k):
        Nk = np.sum(gamma[:, i])
        mju[i] = np.dot(gamma[:, i], X) / Nk
        dis = X - mju[i]  # row vector matrix
        sigma[i] = np.dot(gamma[:, i] * dis.T, dis) / Nk
        pre[i] = Nk/n


def main():
    global X, n, k, pre, mju, sigma, gamma
    data = np.loadtxt('wine.csv', delimiter=',')
    parser = argparse.ArgumentParser(
        description='EM clustering program from UCI dataset.')
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
        if abs(like - last_like) <= 1e-4:
            return step, like
        else:
            last_like = like


if __name__ == '__main__':
    main()
