from sklearn.datasets import make_blobs
from sklearn import tree
from dtreeviz.trees import *
import pandas as pd



X, y = make_blobs(n_samples=300, centers=4,
                  random_state=0, cluster_std=1.0)
X = pd.DataFrame(X)
X.columns = ['x_1', 'x_2']
y = pd.DataFrame(y)
y.columns = ['y']
class_names = np.unique(y.values)
y = y['y'].map({n:i for i, n in enumerate(class_names)})

## colors
base = plt.cm.get_cmap('gist_rainbow')
color_list = base(np.linspace(0, 1, 4))
color_list = color_list * 255
color_list = color_list.astype('int')
new_list = ["#{0:02x}{1:02x}{2:02x}".format(
    color_list[i][0], color_list[i][1], color_list[i][2]) for i in range(len(color_list[0]))]


my_colors = [None] + [new_list[0:i+1] for i in range(len(new_list))]
from dtreeviz.colors import adjust_colors
colors = adjust_colors(None)
colors['classes'] = my_colors


plt.close('all')
fig, ax = plt.subplots(2, 2, figsize=(10, 12))
#fig.subplots_adjust(left=0.02, right=0.98, top=0.3,bottom=0.1, wspace=0.1)


for axi, depth  in zip(ax.reshape(-1,).tolist(), range(1, 5)):
    ct = ctreeviz_bivar(axi, X , y, max_depth=depth,
                    feature_names = ['x_1', 'x_2'],
                    class_names=['class1', 'class2', 'class3', 'class4'],
                    target_name='y',
                    fontsize=12,
                    show={'splits','title'},
                    colors = colors)
plt.tight_layout(pad = 1)    
plt.show()




fig, ax = plt.subplots(2, 2, figsize=(10, 12))
plt.close('all')

fig = plt.figure()
#fig.subplots_adjust(left=0.02, right=0.98, top=0.3,bottom=0.1, wspace=0.1)

for xx, yy, depth in zip([1, 2, 1, 2], [1, 1, 2, 2],range(1,5)):
    fig.add_subplot(xx, yy ,depth)

plt.tight_layout(pad = 1)    
plt.show()



fig = plt.figure()
fig.add_subplot(1, 1, 1)

clsfr = tree.DecisionTreeClassifier(max_depth = depth)
clsfr.fit(X.values, y.values)
viz = dtreeviz(clsfr, X, y, target_name='y',
                   feature_names = ['x_1', 'x_2'],
                   class_names=['class1', 'class2', 'class3', 'class4']
                   )
viz.view()




######################################################################
## random forest
######################################################################
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from matplotlib.patches import Polygon
from itertools import chain
from sklearn.datasets import make_blobs
from sklearn.ensemble import RandomForestClassifier
import matplotlib.animation as animation

X, y = make_blobs(n_samples=300, centers=4,
                  random_state=0, cluster_std=1.0)

plot_step = 0.02 

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1 
xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                     np.arange(y_min, y_max, plot_step)) 
n_estimators = 31 
  
mycmap=plt.cm.Paired
mycmap=plt.cm.Paired
colors = [mycmap(1), mycmap(3), mycmap(6), mycmap(9)]
base = plt.cm.get_cmap('gist_rainbow')
cmap = plt.cm.Reds
redc = (0.99, 0.96078431372549022, 0.94117647058823528, 0.1)

idx0 = np.where(y==0)[0]
idx1 = np.where(y==1)[0]
idx2 = np.where(y==2)[0]
idx3 = np.where(y==3)[0]


fig, ax = plt.subplots(1)
ax.scatter(X[:, 0], X[:, 1], c=y,  cmap=base, edgecolors = 'k')

def init():
    return []
         
clf = RandomForestClassifier(n_estimators = n_estimators,max_depth=3, warm_start=True)

def run(j):
    if j > n_estimators:
        return
    clf.set_params(n_estimators=j)
    clf.fit(X, y)    
    Z = clf.estimators_[j-1].predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    ax.scatter(X[:, 0], X[:, 1], c = y, cmap=base, edgecolors = 'k')
    
    cont = ax.contourf(xx, yy, Z, cmap = base, alpha=0.1)

    return [cont]

ani = animation.FuncAnimation(fig, func = run, init_func = init, frames = chain((i + 1 for i in np.arange(0, n_estimators)), (n_estimators +1 for i in np.arange(0, 20))),
                       interval = 300,  blit = False) 
ani.save('/home/martin/python/random_forest.gif', writer = 'imagemagick', fps = 2)

