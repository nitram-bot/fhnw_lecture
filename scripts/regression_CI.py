import statsmodels.api as sm


# data example
y = np.load('/home/martin/python/fhnw_lecture/scripts/regression_y.pickle.npy')
X = np.load('/home/martin/python/fhnw_lecture/scripts/regression_X.pickle.npy')

# the x (small x) is just for plotting purpose
x = np.arange(1, 12, 0.05).reshape((-1, 1))
x_intercept = np.c_[np.ones(x.shape[0]), x]


X_intercept = np.c_[np.ones(X.shape[0]), X]

ols_result_lin = sm.OLS(y, X_intercept).fit()
y_hat_lin = ols_result_lin.get_prediction(x_intercept)
print(ols_result_lin.summary()) # beta-coefficients

dt_lin = y_hat_lin.summary_frame()
mean_lin = dt_lin['mean']
meanCIs_lin = dt_lin[['mean_ci_lower', 'mean_ci_upper']]
obsCIs_lin = dt_lin[['obs_ci_lower', 'obs_ci_upper']]

### figure for linear plot
f = plt.figure(figsize=(5, 5), dpi=100)
plt.title(label='linear regression', fontdict={'fontsize':20})
axes = f.add_subplot(111)

axes.plot(X_intercept[:,1], y, 'ro')
axes.plot(x_intercept[:, 1], mean_lin.values.reshape((-1,)), color = "red", label = "regression line")
axes.plot(x_intercept[:, 1], obsCIs_lin.iloc[:, 0], color = "darkgreen", linestyle = "--", 
         label = "Observations CI")
axes.plot(x_intercept[:, 1], obsCIs_lin.iloc[:, 1], color = "darkgreen", linestyle = "--")

axes.plot(x_intercept[:, 1], meanCIs_lin.iloc[:, 0], color = "blue", linestyle = "--", 
         label = "Mean Prediction CI")
axes.plot(x_intercept[:, 1], meanCIs_lin.iloc[:, 1], color = "blue", linestyle = "--")
axes.legend()

axes.set_ylim([np.min(y)-10, np.max(y) +10])

plt.close('all')



# for quadratic term
X_intercept_quad = np.c_[X_intercept, X**2]

# for plotting:
x = np.arange(1, 12, 0.05).reshape((-1, 1))
x_intercept_quad = np.c_[np.ones(x.shape[0]), x, x**2]

ols_result_quad = sm.OLS(y, X_intercept_quad).fit()
print(ols_result_quad.summary())

y_hat_quad = ols_result_quad.get_prediction(x_intercept_quad)
dt_quad = y_hat_quad.summary_frame()
mean_quad = dt_quad['mean']
meanCIs_quad = dt_quad[['mean_ci_lower', 'mean_ci_upper']]
obsCIs_quad = dt_quad[['obs_ci_lower', 'obs_ci_upper']]


### figure for linear plot
f = plt.figure(figsize=(5, 5), dpi=100)
plt.title(label='regression with quadratic term', fontdict={'fontsize':20})
axes = f.add_subplot(111)

axes.plot(X_intercept_quad[:,1], y, 'ro')
axes.plot(x_intercept_quad[:, 1], mean_quad.values.reshape((-1,)), color = "red", label = "regression line")
axes.plot(x_intercept_quad[:, 1], obsCIs_quad.iloc[:, 0], color = "darkgreen", linestyle = "--", 
         label = "Observations CI")
axes.plot(x_intercept_quad[:, 1], obsCIs_quad.iloc[:, 1], color = "darkgreen", linestyle = "--")

axes.plot(x_intercept_quad[:, 1], meanCIs_quad.iloc[:, 0], color = "blue", linestyle = "--", 
         label = "Mean Prediction CI")
axes.plot(x_intercept[:, 1], meanCIs_quad.iloc[:, 1], color = "blue", linestyle = "--")
axes.legend()

axes.set_ylim([np.min(y)-10, np.max(y) +10])

plt.close('all')



## bootstrapping as a robust method
# data example
from sklearn.linear_model import ElasticNet
from random import choices
from sklearn.linear_model import Lasso

y = np.load('/home/martin/python/fhnw_lecture/scripts/regression_y.pickle.npy')
X = np.load('/home/martin/python/fhnw_lecture/scripts/regression_X.pickle.npy')


X = np.c_[np.ones(X.shape[0]), X, X**2, X**3, X**4]
x = np.arange(1, 12, 0.05).reshape((-1, 1))
x = np.c_[np.ones(x.shape[0]), x, x**2, x**3, x**4]
indices = np.arange(0, X.shape[0])

drew = choices(indices, k=len(indices))

prediction = Lasso(alpha=2, fit_intercept=True).fit(X[drew,:], y[drew, :]).predict(x)

                                                          
