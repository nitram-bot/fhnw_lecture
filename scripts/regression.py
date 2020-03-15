from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

#length = 11
#X = np.arange(1, 12).reshape((11,1))
#y = X + 10 * np.random.rand(11).reshape((11, 1))
#y = y + 0.3 * y**2
#y = y - np.min(y)
#y = np.array([6, 12, 7, 12, 11, 10, 16, 22, 19, 27, 23]).reshape(11, 1)

np.save('/home/martin/python/fhnw_lecture/scripts/regression_y.pickle', y,\
        allow_pickle=True)
np.save('/home/martin/python/fhnw_lecture/scripts/regression_X.pickle', X,\
        allow_pickle=True)

np.savetxt('/home/martin/python/fhnw_lecture/scripts/regression_y.csv', y,\
        delimiter=",")
np.savetxt('/home/martin/python/fhnw_lecture/scripts/regression_X.csv', X,\
        delimiter=",")
y = np.load('/home/martin/python/fhnw_lecture/scripts/regression_y.pickle.npy')
X = np.load('/home/martin/python/fhnw_lecture/scripts/regression_X.pickle.npy')
model = LinearRegression()
model.fit(X, y)
y_hat = model.coef_ * X + model.intercept_
# plt.plot(X, y, 'ro')
# plt.show()
f = plt.figure(figsize=(8, 8), dpi=100)
plt.title(label='regression line, residues', fontdict={'fontsize':20})
axes = f.add_subplot(111)
axes.title(label='regression line, residues')
axes.plot(X, y, 'ro', X, y_hat)
#axes = plt.gca()
axes.set_ylim([np.min(y)-5, np.max(y) +5])
for i in range(len(y)):
    plt.plot((X[i, 0], X[i, 0]), (y[i], y_hat[i]))

axes.set_xlabel('X') 
axes.set_ylabel('Y')

axes.annotate('$y$', xy=(X[-3, 0], y[-3, 0]), xycoords='data',
            xytext=(X[-3, 0] - 1.5, y[-3, 0] + 1), textcoords='data',
            size = 20, arrowprops=dict(arrowstyle="->"))

axes.annotate('$\hat{y}$', xy=(X[-3, 0], y_hat[-3, 0]), xycoords='data',
            xytext=(X[-3, 0] - 1.5, y_hat[-3, 0] + 1), textcoords='data',
            size = 20, arrowprops=dict(arrowstyle="->"))

axes.annotate('$\hat{y} = a + bX$', xy=(X[3, 0] + 0.5, model.coef_ * (X[3, 0] + 0.5) + model.intercept_),
              xycoords='data', xytext=(X[3, 0] + 0.5, 55), textcoords='data',
              horizontalalignment = 'center',
              size = 20, arrowprops=dict(arrowstyle="->"))
plt.show()

plt.close('all')




###################################
from numpy.linalg import inv
# polynomial
y = np.load('/home/martin/python/fhnw_lecture/scripts/regression_y.pickle.npy')
X = np.load('/home/martin/python/fhnw_lecture/scripts/regression_X.pickle.npy')
# underdetermined, ill-posed: infinitely many solutions
X = np.c_[X, X**2, X**3, X**4, X**5, X**6, X**7, X**8, X**9, X**10, X**11, X**12, X**13]
x = np.arange(1, 12, 0.05).reshape((-1, 1))
x = np.c_[x, x**2, x**3, x**4, x**5, x**6, x**7, x**8, x**9, x**10, x**11, x**12, x**13]
model.fit(X, y)
y_hat = np.dot(x , model.coef_.T)  + model.intercept_
# plt.plot(X, y, 'ro')
# plt.show()

# because numpy is using LU decomposition for matrix inversion
coefs = np.dot(np.dot(inv(np.dot(X.T,X)),X.T),y)

# the only thing to proof that sth. is wrong is to use the computed
# inverse and solve for (X.T * X): (X.T * X)^-1 * (X.T * T) = np.eye()
In [206]: solve(inv(np.dot(X.T, X)), np.eye(13))
/home/martin/anaconda3/bin/ipython:1: LinAlgWarning: Ill-conditioned matrix (rcond=3.85425e-21): result may not be accurate.
# show in R that it is not working





f = plt.figure(figsize=(8, 8), dpi=100)
plt.title(label='regression line, residues', fontdict={'fontsize':20})
axes = f.add_subplot(111)

axes.plot(X[:,0], y, 'ro', x[:,0], y_hat.reshape((-1,)))
#axes = plt.gca()
axes.set_ylim([np.min(y)-5, np.max(y) +5])

