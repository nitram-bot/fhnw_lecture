#%matplotlib inline
import matplotlib.pyplot as plt
import seaborn; seaborn.set()  # for plot styling
import numpy as np
from ipywidgets import interact
from sklearn.metrics import pairwise_distances_argmin
from sklearn.neighbors import NearestCentroid
from sklearn.datasets.samples_generator import make_blobs
import matplotlib.gridspec as gridspec
from sklearn.cluster import KMeans
from itertools import chain
import matplotlib.animation as animation    
from IPython.display import HTML
from scipy.spatial import distance

# make data
X, y = make_blobs(n_samples=300, centers=4,
                      random_state=0, cluster_std=0.60)


# initialize centers
centers = np.array([[min(X[:, 0]) + np.random.rand(1)[0], min(X[:, 1]) + np.random.rand(1)[0]],
                    [min(X[:, 0]) + np.random.rand(1)[0], min(X[:, 1]) + np.random.rand(1)[0]], 
                    [min(X[:, 0]) + np.random.rand(1)[0], min(X[:, 1]) + np.random.rand(1)[0]], 
                    [min(X[:, 0]) + np.random.rand(1)[0], min(X[:, 1]) + np.random.rand(1)[0]]])

clf = NearestCentroid()
clf.fit(centers, np.array([0, 1, 2, 3]))
labels = clf.predict(X)
inertia = [np.sum([distance.euclidean(x , centers[i]) for i in np.unique(labels) for x in X[labels == i]])]

km = KMeans(n_clusters = 4,\
            init = centers,\
            n_init=1, max_iter=1).fit(X)

centers = [centers]
labels = [labels]
while inertia[-1] != km.inertia_:
    inertia.append(km.inertia_)
    centers.append(km.cluster_centers_)
    labels.append(km.labels_)
    km = KMeans(n_clusters = 4,\
            init = km.cluster_centers_,\
            n_init=1, max_iter=1).fit(X)
    

fig = plt.figure(figsize=(12, 9))
G = gridspec.GridSpec(1, 3)  
axes_1 = plt.subplot(G[0, 0])
axes_1.set_xlabel('iteration')
axes_1.set_ylabel('sum of squared dists')
axes_1.set_title('reduction in within cluster variance')
axes_1.set_xlim([-0.5, len(labels) + 0.5])
axes_1.set_ylim([min(inertia) -50, max(inertia) + 50])
#pl.xticks(np.arange(0, n_estimators, 1.0))
axes_2 = plt.subplot(G[0, 1:3])
axes_2.set_xlim([min(X[:,0]) - 0.2, max(X[:, 0]) + 0.2])
axes_2.set_ylim([min(X[:,1])- 0.2, max(X[:, 1]) + 0.2])
mycmap=plt.cm.Paired
colors = [np.array([mycmap(1)]), np.array([mycmap(10)]), np.array([mycmap(2)]), np.array([mycmap(20)])]


#plot_step_of_k_means(labels[0], centers[0])

# muss mit 1 starten
def run(j):
    idx0 = np.where(labels[j]== 0)[0]
    idx1 = np.where(labels[j]== 1)[0]
    idx2 = np.where(labels[j]== 2)[0]
    idx3 = np.where(labels[j]== 3)[0]

    axes_2.scatter(X[idx0, 0], X[idx0,1], marker = 'x', c=colors[0], edgecolors = colors[0])
    axes_2.scatter(X[idx1, 0], X[idx1,1], marker = 'x', c=colors[1], edgecolors = colors[1])
    axes_2.scatter(X[idx2, 0], X[idx2,1], marker = 'x', c=colors[2], edgecolors = colors[2])
    axes_2.scatter(X[idx3, 0], X[idx3,1], marker = 'x', c=colors[3], edgecolors = colors[3])

    if j == 0:
        axes_2.scatter(centers[j][:, 0], centers[j][:, 1], marker= 'o',\
                       c = np.array(colors).reshape((4, 4)), edgecolors = 'blue', s=80)
        axes_1.plot([0, inertia[j]], 'o')
    else:
        axes_1.plot([j-1, j], [inertia[j-1], inertia[j]], '-bo')
        for i in range(len(colors)):
            axes_2.plot([centers[j-1][i][0], centers[j][i][0]],\
                        [centers[j-1][i][1], centers[j][i][1]], '-bo',\
                        color = colors[i][0])
        axes_2.scatter(centers[j][:, 0], centers[j][:, 1], marker= 'o',\
                       c = np.array(colors).reshape((4, 4)), edgecolors = 'blue', s=80)



def init():
    return[]
    
ani = animation.FuncAnimation(fig, func = run, init_func = init, frames = np.arange(1, len(labels)),
                       interval = 200,  blit = False)

ani.save('/home/martin/k-means.gif', writer = 'imagemagick', fps = 2)
