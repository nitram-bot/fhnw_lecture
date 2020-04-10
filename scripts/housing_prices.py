# box cox transform
from scipy.stats import boxcox
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from random import choices
# read in data
train = pd.read_csv('../data/train.csv')
test = pd.read_csv('../data/test.csv')

train_ID = train['Id']
test_ID = test['Id']

train.drop('Id', axis = 1, inplace = True)
test.drop('Id', axis = 1, inplace = True)

SalePrice = train['SalePrice']
train.drop('SalePrice', axis=1, inplace = True)

data = pd.concat((train, test))
data.reset_index(drop = True, inplace = True)
# categorical and numericalvariables:

categorical = [var for var in data.columns if data[var].dtype=='O']
numerical = [var for var in data.columns if data[var].dtype!='O']

# missing values:
# for categorical data, missing values often is the absence of a feature
data[categorical] = data[categorical].fillna('None')

# for numerical data, we add an extra column indicating missing values

variables_na = []
for val in numerical:
    data[val + '_na'] = pd.isnull(data[val])
    variables_na.append(val + '_na')
    data[val].fillna(data[val].mean(), inplace = True)


# transforming the numerical values to be more normaly distributed
# we add 1 because boxcox for 0 is not possible
#skewed_feats = data[numerical].apply(lambda x: skew(x.dropna())) #compute skewness
#skewed_feats = skewed_feats[skewed_feats > 0.75]
#skewed_feats = skewed_feats.index
#data[skewed_feats] = np.log1p(data[skewed_feats])

#box_cox = []
#not_box_cox = []
#for val in numerical:
#    if not any(data[val] <= 0):
#        new_vals, lamb = boxcox(data[val] + 1)
#        if np.abs(lamb) < 8:
#            data[val + '_box_cox'] = new_vals
#            box_cox.append(val + '_box_cox')
#        else:
#            not_box_cox.append(val)


box_cox = []
for val in numerical:
    new_vals, lamb = boxcox(data[val] + 1)
    if np.abs(lamb) < 8:
        data[val + '_box_cox'] = new_vals
        box_cox.append(val)


## no mean-encoding so far
data_base = pd.get_dummies(data[[col for col in data.columns if col not in variables_na]])
data_na = pd.get_dummies(data[variables_na])


## interaction terms
## temporarily we need numbers for categorical values
#for val in categorical:
#    if not val.endswith('_na'):
#        val_dict = {k:i for i,k in enumerate(data[val].unique())}
#        data[val] = data[val].map(val_dict)

data_base = data_base.astype(np.float32)
data_na = data_na.astype(np.float32)

data = pd.concat([data_base, data_na], axis = 1)
for col in data.columns:
    if np.isinf(data[col]).any():
        print(f'this variable: {col}')
        
gbm = GradientBoostingRegressor(n_estimators = 32, max_depth = 4)
gbm.fit(data[: len(train_ID)].values, SalePrice.values)


## this works pretty good but is an advanced topic:
# leaves = pd.DataFrame(gbm.apply(data.values)).astype('category')
# data = pd.concat([data, pd.get_dummies(leaves)], axis = 1)

indizes = np.argsort(gbm.feature_importances_)
from itertools import combinations

interactions = []
for comb in list(combinations(data.columns[indizes[-55:]], 2)):
    data[comb[0] + '_x_' + comb[1]] = data[comb[0]] * data[comb[1]]
    interactions.append(comb[0] + '_x_' + comb[1])

data_interactions = data[interactions]    


##########
# box_cox transformed are removed
base = data_base[[col for col in data_base.columns if not col.endswith('_box_cox')]]
# box_cox is admitted; original variables removed
with_box_cox = data_base[[col for col in data_base.columns if not col in box_cox]]
with_na = pd.concat([with_box_cox, data_na], axis = 1)
with_interactions = pd.concat([with_na, data_interactions], axis = 1)
y = np.log1p(SalePrice)
# lambda parameter for total penalty
lamb = 10**(np.linspace(-1, 0.2, 15))
# ratio
ratio = np.linspace(0, 1, 10)

error = []
best_parameters = []
for d in [base, with_box_cox, with_na, with_interactions]:
    # scale variables
    scaler = StandardScaler()
    scaler.fit(d) #  fit  the scale        

    X_train = scaler.transform(d[:len(train_ID)])
    #test = scaler.transform(data[len(train_ID):])
    
    get_results = [(l, r, np.mean(np.sqrt(-cross_val_score(ElasticNet(alpha = l,
                                                            l1_ratio = r),
            X_train, y , scoring = 'neg_mean_squared_error',
            cv = 5, n_jobs = -1))))
                for l in lamb for r in ratio]

    least_error = np.min([i[2] for i in get_results])
    error.append(least_error)
    parameters = [i[0:2] for i in get_results if i[2] == least_error]
    best_parameters.append(parameters)
    print(f'least error is: {least_error}, best parameters are: {parameters}')


from sklearn.model_selection import KFold # import KFold
import warnings
warnings.filterwarnings('ignore')

kf = KFold(n_splits = 5, shuffle=True)
X = with_interactions[:len(train_ID)]
CIS = np.empty((2, X.shape[0]))
y_hat = np.empty((y.shape[0],))
for train_index, test_index in kf.split(X):
    X_train = scaler.transform(X)[train_index,:]
    y_train = y[train_index]
    X_test = scaler.transform(X)[test_index, :]
    y_test = y[test_index]
    indices = np.arange(0, X_train.shape[0])

    sampler = (choices(indices, k = len(indices)) for i in range(200))
    CIS[:, test_index] = np.percentile(
        np.array(
            [
             Ridge(alpha=best_parameters[-1][0][0], fit_intercept=True)\
             .fit(X_train[drew,:], y_train.values[drew])\
                              .predict(X_test).tolist()
             for drew in sampler]
             ), [2.5, 97.5], axis = 0)


    y_hat[test_index] = Ridge(alpha=best_parameters[-1][0][0], fit_intercept=True).fit(X_train, y_train.values)\
                              .predict(X_test)



y_hat_original_scale = exp(y_hat) + 1






# ratio
ratio = np.linspace(0, 0.2, 10)
get_results = [(l, r, np.mean(np.sqrt(-cross_val_score(ElasticNet(alpha = l,
                                                          l1_ratio = r),
           X_train, y , scoring = 'neg_mean_squared_error',
           cv = 5, n_jobs = -1))))
               for l in lamb for r in ratio]


model = ElasticNet(alpha = parameters[0][0], l1_ratio =parameters[0][1])
model.fit(train, SalePrice)

model.predict(test)




### other solution
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib

import matplotlib.pyplot as plt
from scipy.stats import skew
from scipy.stats.stats import pearsonr

train = pd.read_csv('../data/train.csv')
test = pd.read_csv('../data/test.csv')

all_data = pd.concat((train.loc[:,'MSSubClass':'SaleCondition'],
                      test.loc[:,'MSSubClass':'SaleCondition']))

train["SalePrice"] = np.log1p(train["SalePrice"])

#log transform skewed numeric features:
numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index

skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness
skewed_feats = skewed_feats[skewed_feats > 0.75]
skewed_feats = skewed_feats.index

all_data[skewed_feats] = np.log1p(all_data[skewed_feats])
all_data = pd.get_dummies(all_data)
all_data = all_data.fillna(all_data.mean())
X_train = all_data[:train.shape[0]]
X_test = all_data[train.shape[0]:]
y = train.SalePrice

def rmse_cv(model):
    rmse= np.sqrt(-cross_val_score(model, X_train, y, scoring="neg_mean_squared_error", cv = 5))
    return(rmse)

from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso

alphas = [0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75]
cv_ridge = [rmse_cv(Ridge(alpha = alpha)).mean() 
            for alpha in alphas]    

cv_lasso = [rmse_cv(Lasso(alpha = alpha)).mean() 
            for alpha in alphas]    



cv_ela = [rmse_cv(ElasticNet(alpha = alpha, l1_ratio=r)).mean() 
            for alpha in alphas for r in ratio]    
