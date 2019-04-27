# rndgen.py 随机数生成器
import argparse
import csv

import numpy.random as npr


def main():
    parser = argparse.ArgumentParser(
        description='Generate random (X1,X2,Y) tuples for K-mean clustering.')
    parser.add_argument('-n', type=int, default=10,
                        help='The number of random tuples each type. Default is 10.')
    args = parser.parse_args()
    if args.n <= 0:
        parser.error('Expect n to be a positive integer')
    # generate random number pairs
    x1 = [npr.normal(loc=[2.0, 3.0], scale=1.0) for i in range(args.n)]
    x2 = [npr.normal(loc=[-1.0, 1.0], scale=1.0) for i in range(args.n)]
    x3 = [npr.normal(loc=[3.0, -1.0], scale=1.0) for i in range(args.n)]
    # store the data to a file
    with open('rawdata.csv', 'w', newline='') as csvfile:
        cwrite = csv.writer(csvfile)
        for xi in x1:
            cwrite.writerow([str(xi[0]), str(xi[1]), "1"])
        for xj in x2:
            cwrite.writerow([str(xj[0]), str(xj[1]), "0"])
        for xk in x3:
            cwrite.writerow([str(xk[0]), str(xk[1]), "2"])


if __name__ == '__main__':
    main()
