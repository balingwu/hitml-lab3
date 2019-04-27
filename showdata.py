# showdata.py 简单的可视化辅助工具
from numpy import loadtxt, where
from pylab import scatter, show, legend, xlabel, ylabel
 
#load the dataset
data = loadtxt('rawdata.csv', delimiter=',')
 
X = data[:, 0:2]
y = data[:, 2]
 
pos = where(y == 1)
neg = where(y == 0)
neu = where(y == 2)
scatter(X[pos, 0], X[pos, 1], marker='o', c='b')
scatter(X[neg, 0], X[neg, 1], marker='x', c='r')
scatter(X[neu, 0], X[neu, 1], marker='*', c='k')
xlabel('Feature1')
ylabel('Feature2')
legend(['y=1', 'y=0', 'y=2'])
show()