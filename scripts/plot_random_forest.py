#!/usr/bin/env python
# -*- coding: utf-8 -*-


import pylab as pl
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from random import randint
from matplotlib.colors import ListedColormap
from sklearn.base import clone 
import matplotlib.animation as animation    
   
iris = load_iris() 
  

X = iris.data[:, [0, 1]]
#y = np.zeros((150,))
y = iris.target
#y[[randint(0, y.shape[0] -1) for i in  np.arange(0, 4)]] = 1

# Shuffle
idx = np.arange(X.shape[0])
np.random.seed(13)
np.random.shuffle(idx)
X = X[idx]
y = y[idx]

# Standardize
mean = X.mean(axis=0)
std = X.std(axis=0)
X = (X - mean) / std
    
# train model

plot_step = 0.02

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1 
xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                     np.arange(y_min, y_max, plot_step))


n_estimators = 50
clf = RandomForestClassifier(n_estimators = n_estimators).fit(X, y)
#clf = DecisionTreeClassifier().fit(X, y)
#clf = AdaBoostClassifier(n_estimators = 40, learning_rate = 0.1).fit(X, y)
#clf = GradientBoostingClassifier(n_estimators = 100, learning_rate = 0.1).fit(X, y)
cm = plt.cm.Pastel1
cm_bright = ListedColormap(['#FF0000', '#0000FF'])


if isinstance(clf, DecisionTreeClassifier):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    cs = pl.contourf(xx, yy, Z, cmap = plt.cm.Reds)
else:
    for tree in clf.estimators_:
        Z = tree.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        cs = pl.contourf(xx, yy, Z, cmap = plt.cm.Reds, alpha=0.1)

pl.axis("tight")

# Plot the training points
#for i, c in zip(range(2), cm_bright):
for i in range(3):
    idx = np.where(y == i)
    pl.scatter(X[idx, 0], X[idx, 1], cmap=plt.cm.Paired, edgecolors = 'k')
 

pl.axis("tight")     
        
 
  
##--------------------------------------------------------------------------   
## hier speziell für random-forest and tree by tree
##--------------------------------------------------------------------------   
iris = load_iris()  
X = iris.data[:, [0, 1]]
#y = np.zeros((150,))

#y[[randint(0, y.shape[0] -1) for i in  np.arange(0, 4)]] = 1
rands = [randint(0, X.shape[0] -1) for i in np.arange(0, 20)]
positiv_index = [i for i in np.arange(0, X.shape[0]) if i not in set(rands)]
y = iris.target[positiv_index] 
X = iris.data[np.array(positiv_index), 0:2] 

y_hold_out = iris.target[rands]
X_hold_out = iris.data[rands, 0:2]
  
mean = np.vstack((X,X_hold_out)).mean(axis=0)
std = np.vstack((X,X_hold_out)).std(axis=0)
X = (X - mean) / std
X_hold_out = (X_hold_out - mean) / std  
   
# train model

plot_step = 0.02 

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1 
xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                     np.arange(y_min, y_max, plot_step)) 

# Shuffle
idx = np.arange(X.shape[0])
np.random.seed(13)
np.random.shuffle(idx)
X = X[idx]
y = y[idx] 
   

n_estimators = 30
fig = plt.figure(figsize = (6, 6))
ax = plt.axes(xlim = (-3, 3), ylim = (-3, 4))
mycmap=plt.cm.Paired
colors = [mycmap(0), mycmap(1), mycmap(2)]
cmap = plt.cm.Reds
redc = (0.99, 0.96078431372549022, 0.94117647058823528, 0.1)
for i in range(3):
    idx = np.where(y == i)
    idy = np.where(y_hold_out == i)
    plt.scatter(X[idx, 0], X[idx, 1], c=colors[i], edgecolors = 'k')
    plt.scatter(X_hold_out[idy, 0], X_hold_out[idy, 1], c= 'r', edgecolors = 'k')
        
def init():
    return []
       
clf = RandomForestClassifier(n_estimators = n_estimators, warm_start=True) 
def run(i):
    clf.set_params(n_estimators=i)
    clf.fit(X, y)    
    Z = clf.estimators_[i-1].predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    for i in range(3):
        idx = np.where(y == i)
        idy = np.where(y_hold_out == i)
        plt.scatter(X[idx, 0], X[idx, 1], c = colors[i], edgecolors = 'k')
    cont = plt.contourf(xx, yy, Z, cmap = plt.cm.Reds, alpha=0.1)
    return [cont]
       
  
      
   
ani = animation.FuncAnimation(fig, func = run, init_func = init, frames = (i + 1 for i in np.arange(0, 21)),
                       interval = 100,  blit = False) 
                            
ani.save('/home/docker/MISC/animation.gif', writer='imagemagick', fps= 30)



## ---------------------------------------------------------------------
## hier speziell für subplots
## ---------------------------------------------------------------------

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as pl

iris = load_iris()  
X = iris.data[:, [0, 1]]
#y = np.zeros((150,))

#y[[randint(0, y.shape[0] -1) for i in  np.arange(0, 4)]] = 1
rands = [randint(0, X.shape[0] -1) for i in np.arange(0, 20)]
positiv_index = [i for i in np.arange(0, X.shape[0]) if i not in set(rands)]
y = iris.target[positiv_index] 
X = iris.data[np.array(positiv_index), 0:2] 

y_hold_out = iris.target[rands]
X_hold_out = iris.data[rands, 0:2]
  
mean = np.vstack((X,X_hold_out)).mean(axis=0)
std = np.vstack((X,X_hold_out)).std(axis=0)
X = (X - mean) / std
X_hold_out = (X_hold_out - mean) / std  
   
# train model

plot_step = 0.02 

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1 
xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                     np.arange(y_min, y_max, plot_step)) 

# Shuffle
idx = np.arange(X.shape[0])
np.random.seed(13)
np.random.shuffle(idx)
X = X[idx]
y = y[idx] 
    

n_estimators = 21
 
fig = plt.figure(figsize=(6, 4))
G = gridspec.GridSpec(3, 2)  
axes_1 = plt.subplot(G[0, 0])
#pl.xticks(np.arange(0, n_estimators, 1.0))
axes_2 = plt.subplot(G[1:3, 0])
axes_3 = plt.subplot(G[0, 1], sharey = axes_1)
axes_4 = plt.subplot(G[1:3, 1], sharex = axes_2, sharey = axes_2) 
  
#fig = plt.figure(figsize = (6, 6))
#ax = plt.axes(xlim = (-3, 3), ylim = (-3, 4))
mycmap=plt.cm.Paired
colors = [mycmap(0), mycmap(1), mycmap(2)]
cmap = plt.cm.Reds
redc = (0.99, 0.96078431372549022, 0.94117647058823528, 0.1)
for i in range(3):
    idx = np.where(y == i)
    idy = np.where(y_hold_out == i)
    #plt.scatter(X[idx, 0], X[idx, 1], c=colors[i], edgecolors = 'k')
    #plt.scatter(X_hold_out[idy, 0], X_hold_out[idy, 1], c= 'r', edgecolors = 'k')
    axes_2.scatter(X[idx, 0], X[idx, 1], c=colors[i], edgecolors = 'k')
    axes_2.scatter(X_hold_out[idy, 0], X_hold_out[idy, 1], c= 'r', edgecolors = 'k')
    axes_4.scatter(X[idx, 0], X[idx, 1], c=colors[i], edgecolors = 'k')
    axes_4.scatter(X_hold_out[idy, 0], X_hold_out[idy, 1], c= 'r', edgecolors = 'k')
    
def init():
    return []
         
clf = RandomForestClassifier(n_estimators = n_estimators, warm_start=True)
last_score1 = 0.5
last_score2 = 0.5
def run(j):
    global last_score1, last_score2
    clf.set_params(n_estimators=j)
    clf.fit(X, y)    
    Z = clf.estimators_[j-1].predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    for i in range(3):
        idx = np.where(y == i)
        idy = np.where(y_hold_out == i)
        axes_2.scatter(X[idx, 0], X[idx, 1], c = colors[i], edgecolors = 'k')
        axes_2.scatter(X_hold_out[idy, 0], X_hold_out[idy, 1], c = 'r', edgecolors = 'k')
        axes_4.scatter(X[idx, 0], X[idx, 1], c = colors[i], edgecolors = 'k')
        axes_4.scatter(X_hold_out[idy, 0], X_hold_out[idy, 1], c = 'r', edgecolors = 'k')
    cont = axes_2.contourf(xx, yy, Z, cmap = plt.cm.Reds, alpha=0.1)
    axes_4.contourf(xx, yy, Z, cmap = plt.cm.Reds, alpha=0.1)
    axes_1.plot([j-1, j], [last_score1, clf.score(X_hold_out, y_hold_out)], '-bo')
    axes_3.plot([j-1, j], [last_score2, clf.score(X_hold_out, y_hold_out)], '-bo')
    last_score1 = clf.score(X_hold_out, y_hold_out)
    last_score2 = clf.score(X_hold_out, y_hold_out)
    print(last_score)
    return [cont]
 

ani = animation.FuncAnimation(fig, func = run, init_func = init, frames = (i + 1 for i in np.arange(0, n_estimators)),
                       interval = 200,  blit = False) 



   
## MOONS 
## ---------------------------------------------------------------------
## hier speziell für subplots
## ---------------------------------------------------------------------

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from  sklearn.datasets import make_moons  
from  sklearn.datasets import make_circles

Xdata, ydata = make_moons(n_samples = 150, noise = 0.7)
Xdata, ydata = make_circles(n_samples = 150, noise = 0.7)
data = np.random.multivariate_normal(mean = [0, 3], cov = np.array([[0, 1.3],[.9, 0]]), size = 150)


#y = np.zeros((150,))

#y[[randint(0, y.shape[0] -1) for i in  np.arange(0, 4)]] = 1
rands = [randint(0, Xdata.shape[0] -1) for i in np.arange(0, 20)]
positiv_index = [i for i in np.arange(0, Xdata.shape[0]) if i not in set(rands)] 
y = ydata[positiv_index] 
X = Xdata[np.array(positiv_index), 0:2] 

y_hold_out = ydata[rands]
X_hold_out = Xdata[rands, 0:2]

y_hold_out_cor = np.hstack((y[0:int(len(rands)/2)] , y_hold_out[0:int(len(rands)/2)]))
X_hold_out_cor = np.vstack((X[0:int(len(rands)/2)] , X_hold_out[0:int(len(rands)/2)]))
      
mean = np.vstack((X,X_hold_out)).mean(axis=0)
std = np.vstack((X,X_hold_out)).std(axis=0)
X = (X - mean) / std
X_hold_out = (X_hold_out - mean) / std  
    
# train model

plot_step = 0.02 

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1 
xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                     np.arange(y_min, y_max, plot_step)) 

# Shuffle
idx = np.arange(X.shape[0])
np.random.seed(13)
np.random.shuffle(idx)
X = X[idx]
y = y[idx] 
      

n_estimators = 41
 
fig = plt.figure(figsize=(6, 4))
G = gridspec.GridSpec(3, 2)  
axes_1 = plt.subplot(G[0, 0])
#pl.xticks(np.arange(0, n_estimators, 1.0))
axes_2 = plt.subplot(G[1:3, 0])
axes_3 = plt.subplot(G[0, 1], sharey = axes_1)
axes_4 = plt.subplot(G[1:3, 1], sharex = axes_2, sharey = axes_2) 
  
#fig = plt.figure(figsize = (6, 6))
#ax = plt.axes(xlim = (-3, 3), ylim = (-3, 4))
mycmap=plt.cm.Paired
colors = [mycmap(0), mycmap(1), mycmap(2)]
cmap = plt.cm.Reds
redc = (0.99, 0.96078431372549022, 0.94117647058823528, 0.1)
for i in range(3):
    idx = np.where(y == i)
    idy = np.where(y_hold_out == i)
    #plt.scatter(X[idx, 0], X[idx, 1], c=colors[i], edgecolors = 'k')
    #plt.scatter(X_hold_out[idy, 0], X_hold_out[idy, 1], c= 'r', edgecolors = 'k')
    axes_2.scatter(X[idx, 0], X[idx, 1], c=colors[i], edgecolors = 'k')
    axes_2.scatter(X_hold_out[idy, 0], X_hold_out[idy, 1], c= 'r', edgecolors = 'k')
    axes_4.scatter(X[idx, 0], X[idx, 1], c=colors[i], edgecolors = 'k')
    axes_4.scatter(X_hold_out[idy, 0], X_hold_out[idy, 1], c= 'r', edgecolors = 'k')
    
def init():
    return []
         
clf = RandomForestClassifier(n_estimators = n_estimators, warm_start=True)
last_score1 = 0.5
last_score2 = 0.5
def run(j):
    global last_score1
    global last_score2
    clf.set_params(n_estimators=j)
    clf.fit(X, y)    
    Z = clf.estimators_[j-1].predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    for i in range(3):
        idx = np.where(y == i)
        idy = np.where(y_hold_out == i)
        axes_2.scatter(X[idx, 0], X[idx, 1], c = colors[i], edgecolors = 'k')
        axes_2.scatter(X_hold_out[idy, 0], X_hold_out[idy, 1], c = 'r', edgecolors = 'k')
        axes_4.scatter(X[idx, 0], X[idx, 1], c = colors[i], edgecolors = 'k')
        axes_4.scatter(X_hold_out_cor[idy, 0], X_hold_out_cor[idy, 1], c = 'r', edgecolors = 'k')
    cont = axes_2.contourf(xx, yy, Z, cmap = plt.cm.Reds, alpha=0.1)
    axes_4.contourf(xx, yy, Z, cmap = plt.cm.Reds, alpha=0.1)
    axes_1.plot([j-1, j], [last_score1, clf.score(X_hold_out, y_hold_out)], '-bo')
    axes_3.plot([j-1, j], [last_score2, clf.score(X_hold_out_cor, y_hold_out_cor)], '-bo')
    last_score1 = clf.score(X_hold_out, y_hold_out)
    last_score2 = clf.score(X_hold_out_cor, y_hold_out_cor)
    print(last_score1)
    return [cont]
 

ani = animation.FuncAnimation(fig, func = run, init_func = init, frames = (i + 1 for i in np.arange(0, n_estimators)),
                       interval = 200,  blit = False) 
  
        
## RANDOM majority class
## ---------------------------------------------------------------------
## hier speziell für subplots
## ---------------------------------------------------------------------

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics

Xdata = np.random.multivariate_normal(mean = [0, 3], cov = np.array([[1.3, 0 ],[0, .9]]), size = 200)
rands =  unique([randint(0, Xdata.shape[0] -1) for i in np.arange(0, 51)])[0:20]
ydata = np.zeros(Xdata.shape[0])
ydata[rands] = 1
 
skf = StratifiedKFold(n_splits = 5)
probe = skf.split(Xdata, ydata)
train, test = probe.__next__()
  
#y[[randint(0, y.shape[0] -1) for i in  np.arange(0, 4)]] = 1
#rands = [randint(0, Xdata.shape[0] -1) for i in np.arange(0, 20)]
#positiv_index = [i for i in np.arange(0, Xdata.shape[0]) if i not in set(rands)] 
y = ydata[train]
X = Xdata[train, :]
  
y_hold_out = ydata[test]
X_hold_out = Xdata[test, :]
 
cor_inds = np.hstack((test[y_hold_out == 1][0:2], train[y == 1][0:2], test[y_hold_out == 0]))
y_hold_out_cor = ydata[cor_inds]
X_hold_out_cor = Xdata[cor_inds, :]
       
mean = np.vstack((X,X_hold_out)).mean(axis=0)
std = np.vstack((X,X_hold_out)).std(axis=0)
X = (X - mean) / std
X_hold_out = (X_hold_out - mean) / std  
X_hold_out_cor = (X_hold_out_cor - mean) / std     
# train model

plot_step = 0.02 

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1 
xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                     np.arange(y_min, y_max, plot_step)) 

# Shuffle
idx = np.arange(X.shape[0])
np.random.seed(13)
np.random.shuffle(idx)
X = X[idx]
y = y[idx] 
         
  
n_estimators = 31 
  
fig = plt.figure(figsize=(12, 8))
G = gridspec.GridSpec(3, 2)  
axes_1 = plt.subplot(G[0, 0])
#pl.xticks(np.arange(0, n_estimators, 1.0))
axes_2 = plt.subplot(G[1:3, 0])
axes_3 = plt.subplot(G[0, 1], sharey = axes_1)
axes_4 = plt.subplot(G[1:3, 1], sharex = axes_2, sharey = axes_2) 
   
#fig = plt.figure(figsize = (6, 6))
#ax = plt.axes(xlim = (-3, 3), ylim = (-3, 4))
mycmap=plt.cm.Paired
colors = [mycmap(0), mycmap(3), mycmap(5)]
cmap = plt.cm.Reds
redc = (0.99, 0.96078431372549022, 0.94117647058823528, 0.1)
for i in range(2):
    idx = np.where(y == i)
    idy = np.where(y_hold_out == i)
    idy_cor = np.where(y_hold_out_cor == i)
    #plt.scatter(X[idx, 0], X[idx, 1], c=colors[i], edgecolors = 'k')
    #plt.scatter(X_hold_out[idy, 0], X_hold_out[idy, 1], c= 'r', edgecolors = 'k')
    axes_2.scatter(X[idx, 0], X[idx, 1], c=colors[i], edgecolors = 'k')
    axes_2.scatter(X_hold_out[idy, 0], X_hold_out[idy, 1], c= 'r', edgecolors = 'k')
    axes_4.scatter(X[idx, 0], X[idx, 1], c=colors[i], edgecolors = 'k')
    axes_4.scatter(X_hold_out[idy_cor, 0], X_hold_out[idy_cor, 1], c= 'r', edgecolors = 'k')
     
def init():
    return []
         
#clf = RandomForestClassifier(n_estimators = n_estimators, warm_start=True)
clf = AdaBoostClassifier(n_estimators = n_estimators)
clf.fit(X, y)    
staged1 = clf.staged_predict(X_hold_out)
staged2 = clf.staged_predict(X_hold_out_cor) 
last_score1 = 0.5
last_score2 = 0.5
def run(j):
    global last_score1
    global last_score2
    Z = clf.estimators_[j-1].predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    for i in range(2):
        idx = np.where(y == i)
        idy = np.where(y_hold_out == i)
        idy_cor = np.where(y_hold_out_cor == i)
        axes_2.scatter(X[idx, 0], X[idx, 1], c = colors[i], edgecolors = 'k')
        axes_2.scatter(X_hold_out[idy, 0], X_hold_out[idy, 1], c = 'r', edgecolors = 'k')
        axes_4.scatter(X[idx, 0], X[idx, 1], c = colors[i], edgecolors = 'k')
        axes_4.scatter(X_hold_out_cor[idy_cor, 0], X_hold_out_cor[idy_cor, 1], c = 'r', edgecolors = 'k')
    cont = axes_2.contourf(xx, yy, Z, cmap = plt.cm.Reds, alpha=0.1)
    axes_4.contourf(xx, yy, Z, cmap = plt.cm.Reds, alpha=0.1)
    pred1 = staged1.__next__()
    pred2 = staged2.__next__()
    fpr1, tpr1, thresholds1 = metrics.roc_curve(y_hold_out, pred1, pos_label=1)    
    fpr2, tpr2, thresholds2 = metrics.roc_curve(y_hold_out_cor, pred2, pos_label=1)
    actual_score1 = metrics.auc(fpr1, tpr1)    
    actual_score2 = metrics.auc(fpr2, tpr2)
    axes_1.plot([j-1, j], [last_score1, actual_score1], '-bo')
    axes_3.plot([j-1, j], [last_score2, actual_score2], '-bo')
    last_score1 = actual_score1
    last_score2 = actual_score2
    #last_score1 = clf.score(X_hold_out, y_hold_out)
    #last_score2 = clf.score(X_hold_out_cor, y_hold_out_cor)
    print(last_score1)
    return [cont]
 

ani = animation.FuncAnimation(fig, func = run, init_func = init, frames = (i + 1 for i in np.arange(0, n_estimators)),
                       interval = 200,  blit = False) 
          
 
        
   
## RANDOM majority class and RandomForest
## ---------------------------------------------------------------------
## hier speziell für subplots
## ---------------------------------------------------------------------

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from matplotlib.patches import Polygon
from itertools import chain
Xdata = np.random.multivariate_normal(mean = [0, 3], cov = np.array([[1.3, 0.2 ],[0.3, .9]]), size = 180)
rands =  unique([randint(0, Xdata.shape[0] -1) for i in np.arange(0, 51)])[0:20]
ydata = np.zeros(Xdata.shape[0])
ydata[rands] = 1
  
skf = StratifiedKFold(n_splits = 5)
probe = skf.split(Xdata, ydata)
train, test = probe.__next__()
  
#y[[randint(0, y.shape[0] -1) for i in  np.arange(0, 4)]] = 1
#rands = [randint(0, Xdata.shape[0] -1) for i in np.arange(0, 20)]
#positiv_index = [i for i in np.arange(0, Xdata.shape[0]) if i not in set(rands)] 
y = ydata[train]
X = Xdata[train, :]
  
y_hold_out = ydata[test]
X_hold_out = Xdata[test, :]
 
cor_inds = np.hstack((test[y_hold_out == 1][0:2], train[y == 1][0:2], test[y_hold_out == 0]))
y_hold_out_cor = ydata[cor_inds]
X_hold_out_cor = Xdata[cor_inds, :]
       
mean = np.vstack((X,X_hold_out)).mean(axis=0)
std = np.vstack((X,X_hold_out)).std(axis=0)
X = (X - mean) / std
X_hold_out = (X_hold_out - mean) / std  
X_hold_out_cor = (X_hold_out_cor - mean) / std     
# train model

plot_step = 0.02 

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1 
xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                     np.arange(y_min, y_max, plot_step)) 

# Shuffle
idx = np.arange(X.shape[0])
np.random.seed(13)
np.random.shuffle(idx)
X = X[idx]
y = y[idx] 
         
   
n_estimators = 31 
  
fig = plt.figure(figsize=(12, 9))
G = gridspec.GridSpec(3, 2)  
axes_1 = plt.subplot(G[0, 0])
axes_1.set_xlabel('number of trees')
axes_1.set_ylabel('auc')
axes_1.set_title('independent hold out set')
#pl.xticks(np.arange(0, n_estimators, 1.0))
axes_2 = plt.subplot(G[1:3, 0])
axes_3 = plt.subplot(G[0, 1], sharey = axes_1)
axes_3.set_xlabel('number of trees')
axes_3.set_ylabel('auc')
axes_3.set_title('50% known data of minority class in hold out fold')
axes_4 = plt.subplot(G[1:3, 1], sharex = axes_2, sharey = axes_2) 
   
#fig = plt.figure(figsize = (6, 6))
#ax = plt.axes(xlim = (-3, 3), ylim = (-3, 4))
mycmap=plt.cm.Paired
colors = [mycmap(1), mycmap(10), mycmap(2)]
cmap = plt.cm.Reds
redc = (0.99, 0.96078431372549022, 0.94117647058823528, 0.1)

idx0 = np.where(y == 0)[0]
idx1 = np.where(y == 1)[0]
idy0 = np.where(y_hold_out == 0)[0]
idy1 = np.where(y_hold_out == 1)[0]
idy_cor0 = np.where(y_hold_out_cor == 0)[0]
idy_cor1 = np.where(y_hold_out_cor == 1)[0]

l1 = axes_2.scatter(X[idx0, 0], X[idx0, 1], c=colors[0], edgecolors = colors[0])
l2 = axes_2.scatter(X[idx1, 0], X[idx1, 1], c=colors[1], edgecolors = colors[1])
l3 = axes_2.scatter(X_hold_out[idy0, 0], X_hold_out[idy0, 1], c= colors[0], edgecolors = colors[0], marker = '*')
l4 = axes_2.scatter(X_hold_out[idy1, 0], X_hold_out[idy1, 1], c= 'r', edgecolors = 'r', marker = '*')
axes_2.legend((l1, l2, l3, l4), ('train', 'train minority', 'test', 'test minority'), loc='upper right', shadow = False)


r1 = axes_4.scatter(X[idx0, 0], X[idx0, 1], c=colors[0], edgecolors = colors[0])
r2 = axes_4.scatter(X[idx1, 0], X[idx1, 1], c=colors[1], edgecolors = colors[1])
r3 = axes_4.scatter(X_hold_out[idy_cor0, 0], X_hold_out[idy_cor0, 1], c= colors[0], edgecolors = colors[0],
                   marker = '*')
r4 = axes_4.scatter(X_hold_out[idy_cor1, 0], X_hold_out[idy_cor1, 1], c= 'r', edgecolors = 'r',
                   marker = '*')
axes_4.legend((r1, r2, r3, r4), ('train', 'train minority', 'test', 'test minority'), loc='upper right', shadow = False)
     


def init():

    return []
         
clf = RandomForestClassifier(n_estimators = n_estimators, warm_start=True)
last_score1 = 0.5
last_score2 = 0.5
auc_independent = []
auc_dependent = [] 
def run(j):
    global last_score1
    global last_score2
    global auc_independent
    global auc_dependent
    if j > n_estimators:
        return
    clf.set_params(n_estimators=j)
    clf.fit(X, y)    
    Z = clf.estimators_[j-1].predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    for i in range(2):
        idx = np.where(y == i)
        idy = np.where(y_hold_out == i)
        idy_cor = np.where(y_hold_out_cor == i)
        axes_2.scatter(X[idx, 0], X[idx, 1], c = colors[i], edgecolors = colors[i])
        if i == 1:
            axes_2.scatter(X_hold_out[idy, 0], X_hold_out[idy, 1], c = 'r', edgecolors = 'r', marker = '*')
        else:
            axes_2.scatter(X_hold_out[idy, 0], X_hold_out[idy, 1], c = colors[i], edgecolors = colors[i], marker = '*')
            
        axes_4.scatter(X[idx, 0], X[idx, 1], c = colors[i], edgecolors = colors[i], alpha = 1)
        if i == 1:
            axes_4.scatter(X_hold_out_cor[idy_cor, 0], X_hold_out_cor[idy_cor, 1], c = 'r', edgecolors = 'r', alpha = 1,
                       marker = '*')
        else:
            axes_4.scatter(X_hold_out_cor[idy_cor, 0], X_hold_out_cor[idy_cor, 1], c = colors[i], edgecolors = colors[i], alpha = 1,
                       marker = '*')
    cont = axes_2.contourf(xx, yy, Z, cmap = plt.cm.Reds, alpha=0.1)
    axes_4.contourf(xx, yy, Z, cmap = plt.cm.Reds, alpha=0.1)
    pred1 = clf.predict(X_hold_out)
    pred2 = clf.predict(X_hold_out_cor)
    fpr1, tpr1, thresholds1 = metrics.roc_curve(y_hold_out, pred1, pos_label=1)    
    fpr2, tpr2, thresholds2 = metrics.roc_curve(y_hold_out_cor, pred2, pos_label=1)
    actual_score1 = metrics.auc(fpr1, tpr1)    
    actual_score2 = metrics.auc(fpr2, tpr2)
    axes_1.plot([j-1, j], [last_score1, actual_score1], '-bo')
    axes_3.plot([j-1, j], [last_score2, actual_score2], '-bo')
    last_score1 = actual_score1
    last_score2 = actual_score2
    auc_independent.append(actual_score1)
    auc_dependent.append(actual_score2)

    # print overfitting area
    if j == n_estimators:
        max_independent = auc_independent.index(max(auc_independent))
        max_dependent = auc_dependent.index(max(auc_dependent))
        
        poly = matplotlib.patches.Polygon(np.array([[max_independent + 1, 0.2],
                                                    [max_dependent + 1, 0.2],
                                                    [max_dependent + 1, 0.99],
                                                    [max_independent + 1, 0.99]]),
                                              color = 'r', alpha = 0.2)

        axes_3.add_patch(poly)
        axes_3.text(x = (max_independent + ((max_dependent - max_independent)/2)),
                        y = 0.5, s = 'overfitting', color = 'r', alpha = 0.5,
                        horizontalalignment='center')
        
        
    print(last_score1)
    return [cont]
  
 
ani = animation.FuncAnimation(fig, func = run, init_func = init, frames = chain((i + 1 for i in np.arange(0, n_estimators)), (n_estimators +1 for i in np.arange(0, 20))),
                       interval = 200,  blit = False) 
ani.save('/home/docker/MISC/overfit_random_forest.gif', writer = 'imagemagick', fps = 2)
#ani.save('/home/docker/MISC/overfit.mp4', writer = animation.writers['mencoder'], fps = 30, bitrate = 1800)

  
   
    
