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



## test for singular matrix
y = np.load('/home/martin/python/fhnw_lecture/scripts/regression_y.pickle.npy')
X = np.load('/home/martin/python/fhnw_lecture/scripts/regression_X.pickle.npy')

X = np.c_[X, X**2, X**3, X**4, X**5, X**6, X**7]
# for plotting purpose:
x = np.arange(-1, 12, 0.05).reshape((-1, 1))
x = np.c_[x, x**2, x**3, x**4, x**5, x**6, x**7]
from sklearn.linear_model import Ridge
Xc = X - np.mean(X, axis=0)
xc = x - np.mean(x, axis = 0)
yc = y - np.mean(y)
model = Ridge(alpha=2, fit_intercept=False)
model.fit(Xc, yc)

inverse = np.linalg.inv(np.dot(np.transpose(Xc), Xc) + np.eye(Xc.shape[1]) * 2)
Xy = np.dot(np.transpose(Xc),y)

params = np.dot(inverse, Xy)

y_hat = np.dot(xc , model.coef_.T)  + model.intercept_

f = plt.figure(figsize=(5, 5), dpi=100)
plt.title(label='regression line for polynome of 10th degree', fontdict={'fontsize':20})
axes = f.add_subplot(111)

axes.plot(X[:,0], y, 'ro', x[:,0], y_hat.reshape((-1,)) + np.mean(y))
#axes = plt.gca()
axes.set_ylim([np.min(y)-10, np.max(y) +20])

## for comparison
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X, y)
y_overfitted = np.dot(x , model.coef_.T)  + model.intercept_



plt.plot(x[:, 0], y_overfitted, 'g-')


####################################################################
## make contour plots for visualizing l1 and l2 error

X1 = np.random.normal(loc = 1.0, scale = 0.8, size = 100)
X2 = np.random.normal(loc = 0.5, scale = 1.2, size = 100)
beta1 = 1.5
beta2 = 0.5
Y = beta1 * X1 + beta2 * X2
X = np.c_[X1, X2]
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X, Y)
model.intercept_ # essentiall zero
model.coef_ # essentially 0.2 and 0.5

b1 = np.linspace(beta1 - 0.9, beta1 + 0.9, 100)
b2 = np.linspace(beta2 - 0.9, beta2 + 0.9, 100)

bb1, bb2 = np.meshgrid(b1, b2)

Yhat = bb1.reshape(-1, 1) * X1.reshape(1, -1) + bb2.reshape(-1, 1) * X2.reshape(1, -1)
errors = np.square(Yhat - Y.reshape(1, -1))
error = np.sum(errors, axis = 1)/len(Y)
error_to_plot = error.reshape(bb1.shape)


f = plt.figure(figsize=(5, 5), dpi=100)
plt.title(label='minimal errors with penalties', fontdict={'fontsize':13})
axes = f.add_subplot(111)

cp = plt.contour(bb1, bb2, error_to_plot)
plt.clabel(cp, inline=1, fontsize=10)
axes.set_xlabel('b1') 
axes.set_ylabel('b2')
axes.set_ylim([np.min(b2)-0.5, np.max(b2) + 0.5])
axes.set_xlim([np.min(b1)-0.5, np.max(b1) + 0.5])
plt.show()

## next l2 error
constraint_error = 1.0
values = np.linspace(0, 1.0, 100)
constraint_l2 = np.sqrt(constraint_error - values**2)
axes.plot(values, constraint_l2, 'y-', label = 'ridge')
axes.plot(-values, constraint_l2, 'y-')
axes.plot(values, -constraint_l2, 'y-')


constraint_l1 = constraint_error -values
axes.plot(values, constraint_l1, 'r-', label = 'lasso')
axes.plot(-values, constraint_l1, 'r-')
axes.plot(values, -constraint_l1, 'r-')

axes.scatter(beta1, beta2, s = 20)
axes.annotate('$\hat{b}$', xy=(beta1 , beta2 + 0.1), xycoords='data',
              horizontalalignment = 'center', size = 20)             


legs = axes.legend()
# least error for ridge:
Yhat_ridge = np.concatenate((values, values)).reshape(-1,1) * X1.reshape(1, -1) + \
np.concatenate((constraint_l2, -constraint_l2)).reshape(-1,1) * X2.reshape(1, -1)
errors_ridge = np.square(Yhat_ridge - Y.reshape(1, -1))
error_ridge = np.sum(errors_ridge, axis = 1)/len(Y)
index_ridge = np.where(error_ridge ==np.amin(error_ridge))[0][0]
axes.scatter(np.concatenate((values, values))[index_ridge],
             np.concatenate((constraint_l2, -constraint_l2))[index_ridge],
             s=20, c='y')


Yhat_lasso = np.concatenate((values, values)).reshape(-1,1) * X1.reshape(1, -1) + \
np.concatenate((constraint_l1, -constraint_l1)).reshape(-1,1) * X2.reshape(1, -1)
errors_lasso = np.square(Yhat_lasso - Y.reshape(1, -1))
error_lasso = np.sum(errors_lasso, axis = 1)/len(Y)
index_lasso = np.where(error_lasso ==np.amin(error_lasso))[0][0]
axes.scatter(np.concatenate((values, values))[index_lasso],
             np.concatenate((constraint_l1, -constraint_l1))[index_lasso],
             s=20, c='r')



