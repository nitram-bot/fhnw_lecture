#!/usr/bin/env python
# coding: utf-8

# 
# 1. Morning
#     - Data Modelling & Cross Validation
#     - data leakage & dependent data
#     - imbalanced data (example in python)
#     - study: Ebanking Fraud 
#     - Q&A
#     
#     <br>
#     
#     
# 2. Afternoon
#     - Data Basics & historical perspective
#     - Linear Regression
#     - __Trees__
#     - house prices (regression example in python)
#     - Clustering
#     - bonus: Hyperparameter Optimization and AutoML
#     - Q&A

# In[3]:


from IPython.display import Image, display_svg, SVG
import warnings
warnings.filterwarnings('ignore')
# https://sebastianraschka.com/faq/docs/decisiontree-error-vs-entropy.html


# ## Why trees?

# In[4]:


Image("../images/weak_learners_for_boosting.png")
# <img alt="" caption="how data leakage might happen" 
# id="data_leakage" src="../images/weak_learners_for_boosting.png" width="640" height="640">


# image taken form [p.370](https://web.stanford.edu/~hastie/Papers/ESLII.pdf)

# In nachfolgender Graphik wird demonstriert, wie ein decision-tree classifier nach und nach den Input-Variablen-Raum unterteilt um möglichst reine Unterräume zu erhalten. Diese Unterräume entsprechen den jeweiligen Knoten im Baum (nodes), bzw. den Blättern.<br>
# Wichtig ist, dass diese splits auf einer Variablen immer __orthogonal__ sind. 

# In[5]:


from sklearn.datasets import make_blobs
from sklearn import tree
from dtreeviz.trees import *
import graphviz
import pandas as pd


# little hack for adjusting the color of the regions
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
colors['rect_edge'] = '#000000'
colors['edge'] = '#000000'
colors['split_line'] = '#000000'
colors['wedge'] = '#000000'
colors['scatter_edge'] = '#000000'
colors['tesselation_alpha'] = 0.3

X, y = make_blobs(n_samples=300, centers=4,
                  random_state=0, cluster_std=1.0)
X = pd.DataFrame(X)
X.columns = ['x_1', 'x_2']
y = pd.DataFrame(y)
y.columns = ['y']
class_names = np.unique(y.values)
y = y['y'].map({n:i for i, n in enumerate(class_names)})

plt.close('all')
fig, ax = plt.subplots(2, 2, figsize=(10, 8))
# for axi, depth  in zip(ax.reshape(-1,).tolist(), range(1, 5)):
#    ct = ctreeviz_bivar(axi, X , y, max_depth=depth,
#                    feature_names = ['x_1', 'x_2'],
#                    class_names=['class1', 'class2', 'class3', 'class4'],
#                    target_name='y',
#                    fontsize=8,
#                    show={'splits','title'})

for axi, depth in zip(ax.reshape(-1,).tolist(), range(1, 5)):
    dt = tree.DecisionTreeClassifier(max_depth=depth)
    dt.fit(X, y)    
    ct = ctreeviz_bivar(dt, X , y,
                    feature_names = ['x_1', 'x_2'],
                    class_names=['class1', 'class2', 'class3', 'class4'],
                    target_name='y',
                    fontsize=8,
                    ax = axi,
                    show={'splits','title'})
    
plt.tight_layout(pad = 1)    
plt.show()


# Dies ist der decision-trees mit den splits, wie sie oben dargestellt sind.

# In[6]:


get_ipython().system('pip install svglib reportlab')


# In[7]:


from svglib.svglib import svg2rlg
from reportlab.graphics import renderPDF, renderPM

plt.close('all')
#fig, ax = plt.subplots(1,1, figsize=(7, 6))
clsfr = tree.DecisionTreeClassifier(max_depth = 4)
clsfr.fit(X.values, y.values)
viz = dtreeviz(clsfr, X, y, target_name='y',
                   feature_names = ['x_1', 'x_2'],
                   class_names=['class1', 'class2', 'class3', 'class4'],
                   fancy = False
                   )
viz.save('tree.svg')
drawing = svg2rlg("tree.svg")
renderPM.drawToFile(drawing, "tree.png", fmt="PNG")
Image("tree.png", width=420, height=420)


# ### Compare to linear Regression (logistic regression in this case)
# Logistic Regression only works for binary classes. But we can always classify __o__ne class __v__ersus the __r__est (ovr) of the other classes - this allows multiclass classification with logistic regression.<br>
# We can see, whereas the classification tree can approximate __non-linear seperating lines__ with many rectangular splits, the logistic regression can only do linear splits.

# In[8]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression(solver='liblinear', random_state=0, fit_intercept=True)

import matplotlib.pyplot as plt
# plt.figure(figsize=(10, 8), dpi=100)
cols = ['#ff0028', '#5bff00', '#008fff', '#ff00bf']

plt.close('all')
fig = plt.figure(figsize=(6, 9), dpi=100)
ax = fig.add_subplot(111)
# plt.subplots_adjust(bottom=0.05, top=0.9, left=0.05, right=0.95)
# plt.subplot(121)
#fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 8))
# fig, ax = plt.subplots(2, 2, figsize=(10, 8))
# plt.scatter(X.x_1, X.x_2, marker="o", c=y.map({0:cols[0], 1:cols[1], 2:cols[2], 3:cols[3]}).tolist(), s=25, edgecolor="k")
# for i in np.unique(y):
#     y_0 = y == i
#     y_0 = y_0.astype(int)
#     model.fit(X, y_0)
#     b = model.intercept_[0]
#     w1, w2 = model.coef_.T
#     # Calculate the intercept and gradient of the decision boundary.
#     c = -b/w2
#     m = -w1/w2

    # Plot the data and the classification with the decision boundary.
#     xmin, xmax = -5, 5
#     ymin, ymax = -3, 12
#     xd = np.array([xmin, xmax])
#     yd = m*xd + c
#     plt.plot(xd, yd, 'k', lw=1, ls='--')
    
 
dt = tree.DecisionTreeClassifier(max_depth=5)
dt.fit(X, y)
ct = ctreeviz_bivar(dt, X , y,
                    feature_names = ['x_1', 'x_2'],
                    class_names=['class1', 'class2', 'class3', 'class4'],
                    target_name='y',
                    fontsize=8,
                    ax = ax,
                    show={'splits','title'})
ax.scatter(X.x_1, X.x_2, marker="o", c=y.map({0:cols[0], 1:cols[1], 2:cols[2], 3:cols[3]}).tolist(), s=25, edgecolor="k")
for i in np.unique(y):
    y_0 = y == i
    y_0 = y_0.astype(int)
    model.fit(X, y_0)
    b = model.intercept_[0]
    w1, w2 = model.coef_.T
    # Calculate the intercept and gradient of the decision boundary.
    c = -b/w2
    m = -w1/w2

    # Plot the data and the classification with the decision boundary.
    xmin, xmax = -5, 5
    ymin, ymax = -3, 12
    xd = np.array([xmin, xmax])
    yd = m*xd + c
    ax.plot(xd, yd, 'k', lw=1, ls='--')


# ### Splitting criteria
# For most variants of classification trees, there are basically two important splitting statistics:
#   1. Gini Impurity
#   2. Entropy
#   
# __Gini Impurity__
# 
# \begin{equation}
# \text{Gini} = \sum_i^n p_i (1-p_i)
# \end{equation}
# 
# For a binary classification problem (either 0 or 1) the Gini-Index is:<br>
# 
# \begin{equation}
# \text{Gini} = p_1 (1-p_1) + p_2 (1-p_2) 
# \end{equation}
# 
# Here, $p_1$ is the purity of the respective node in the tree.

# In[9]:


Image('../images/tree_splitting_criteria.png')


# At the root of the tree, we have the following impurities for class 1 and class 0:<br>
# \begin{eqnarray*}
# p_{01} =& \frac{40}{120} = 0.3333\\
# p_{00} =& \frac{80}{120} = 0.6666
# \end{eqnarray*}
# The Gini-Impurity at the root is given by:
# \begin{equation}
# \text{Gini}_{0} = p_{01}\cdot (1-p_{01}) + p_{00}\cdot (1-p_{00}) = 0.4444
# \end{equation}
# <br>
# 
# Next we compute the Gini-Impurities $\text{Gini}_1$ and $\text{Gini}_2$ for the child-nodes after the first split:
# \begin{eqnarray*}
# p_{11} =& \frac{28}{70} = 0.4\\
# p_{10} =& \frac{42}{70} = 0.6
# \end{eqnarray*}
# The Gini-Impurity of the first child node is given by:
# \begin{equation}
# \text{Gini}_{1} = p_{11}\cdot (1-p_{11}) + p_{10}\cdot (1-p_{10}) = 0.48
# \end{equation}
# For the second child node, we get:
# \begin{eqnarray*}
# p_{21} =& \frac{12}{50} = 0.24\\
# p_{20} =& \frac{38}{50} = 0.76
# \end{eqnarray*}
# The Gini-Impurity of the second child node is given by:
# \begin{equation}
# \text{Gini}_{2} = p_{21}\cdot (1-p_{21}) + p_{20}\cdot (1-p_{20}) = 0.3648
# \end{equation}

# In the left child node, there are 70 observations, whereas in the right child node, we only have 50 observations. To compute the overall Gini-impurity after the first split, we have the weight the Gini-Impurities of the two child-nodes with the fraction of observations they represent:<br>
# 
# \begin{equation}
# \text{Gini}_{\text{split1}} = \frac{70}{120} \cdot 0.48 + \frac{50}{120} \cdot 0.3658 = 0.4320
# \end{equation}
# <br>
# We see, the impurity could be reduced by the first split from $\,\mathbf{0.4444}\,$ to $\,\mathbf{0.4320}\,$.

# Instead of Gini-Impurity, we could just take classification error as a criterion - this seems most intuitive:<br>
# The classification error in the root node is given by:
# 
# \begin{equation}
# p_0 = \frac{40}{120} = \mathbf{0.333}
# \end{equation}
# The classification error in the first child-node is:<br>
# \begin{equation}
# p_1 = \frac{28}{70} = 0.4
# \end{equation}
# And the classification error in the second child-node is given by: <br>
# \begin{equation}
# p_2 = \frac{12}{50} = 0.24
# \end{equation}
# Now, to compute the reduction in classifiication-error, we have again to weigh the two nodes by the number of observations the contain:<br>
# 
# Classification-error after the first split is:
# \begin{equation}
# \frac{70}{120} \cdot 0.4 + \frac{50}{120} \cdot 0.24 = \mathbf{0.333}
# \end{equation}
# <br>
# Hence, the split is not reducing the classification-error. But without the first split, the following splits that subsequenly lead to zero classification error would no be possible.
# 
# 
# 
# 

# # Genetic Algorithms
# ![](../images/John_Holland.png)

# In[22]:


Image("../seminar_skript/DS_Timeline.png")


# ![](../images/genetic_algorithm_trend.png)

# ### Evolutionary Decision Trees:
# 
# Genetic Algorithms try to mimic the genetic recombination happening in sexual reproduction:
# 
#   - Crossing Over: random recombination between the paired chromosomes inherited from each of one's parents, generally occurring during meiosis (wiki);
#   - fertilization: haploid chromosomes from a random mother and a random father form a new diploid set of chromosomes
#   - mutation: randomly, some genes may change accidentally
#   
#   
# Survival of the Fittest:
# 
#   - only a certain number of the offspring passes the evolutionary bottleneck (the best adapted ones + some randomness)
#   - the survivors form the next parent generation with probability proportional to their fitness
# 
# __Applied to decision trees on can:__
#   - start growing a generation of decision trees, and recombine the fittest trees
#   - grow and prune decision trees with evolutionary principles

# ### my opinion about evolutionary algorithms:
# Before the advent of modern machine-learning algorithms, most algorithms (classification-trees) where optimized in a hill-climbing fashion: __straight to the top__<br>
# But as we all know, the seemingly shortest path is not necessarily the best one.<br>
# The straight path may end, for example, on a steep cliff, i.e the algorithm finds a local optimum that is not identical with the global optimum.<br>
# Evolutionary optimization methods are a way to explore the search space in a more random fashion - avoiding getting stuck in local optima and hopefully finding the local optimum.<br>
# 
# __BUT__: 
#   - There is no guarantee that these algorithms will succeed. 
#   - The search can be very long-lasting.
#   
# __AND__:
#   - Modern Algorithms - as for example Random Forest or Gradient Boosting Trees - have randomness build in.

# ## Random Forest
# Random Forest is an example of classifier Bootstrap Aggregation or bagging.
#  - trees are not very deep (only stumps)
#  - each tree is build on a subsample of data and/or columns -- choosen randomly
#  - results of the individual trees are aggregated (mean)
# 
# ### Pros of Random Forest:
# 
#  - trees can be trained independently: easy to parallelize
#  - classification and regression possible
#  - all other pros of trees like: handling of missing values, insensitive to outliers, numerical and categorical data, etc..
#  - can give a variance estimate (confidence intervals): mean prediction and variance of prediction (see SMAC in AutoML)
#  - averaging allows for arbitrary non-linear relationships
#  
# ### Cons of Random Forest:
#  - black box algorithm: hard to interpret; (see feature importance)

# In[10]:


Image("../images/random_forest.png")
#<img alt="generated with latex" caption="Illustration of random forest" id="random_forest" #src="../images/random_forest.png">


# In[11]:


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
import numpy as np
from IPython.display import HTML
import warnings
warnings.filterwarnings("ignore")

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
         
clf = RandomForestClassifier(n_estimators = 0,max_depth=3, max_samples = 50, warm_start=True)


# In[12]:


def run(j):
    if j > n_estimators:
        return
    clf.set_params(n_estimators=int(j) + 1)
    clf.fit(X, y)    
    Z = clf.estimators_[j].predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    ax.scatter(X[:, 0], X[:, 1], c = y, cmap=base, edgecolors = 'k')
    
    cont = ax.contourf(xx, yy, Z, cmap = base, alpha=0.1)

    return [cont]

ani = animation.FuncAnimation(fig, func = run, init_func = init, frames = list(np.arange(0, n_estimators)) + [n_estimators] * 2,
                       interval = 300,  blit = False) 
HTML(ani.to_jshtml())    


# # Gradient Boosted Trees
# 

# In[13]:


Image("../images/gbm.png", width=820, height=820)
#<img alt="generated with latex" caption="The gradient boosting algorithm" id="gradient_boosting" #src="../images/gbm.png">


# ### Explanation of Gradient Boosted Trees:
#   - we start with a prediction $f_0(x)$, i.e. the mean of $y$ ($\bar{y}$) for regression or the most frequent class in case of classification
#   - the difference between the actual values $y_i$ and our initial start value $f_0(x)$ (residuals) is to be predicted by the first tree $T_1$; the tree uses the variables $\pmb{X}$ to find a rule to group similar residuals in common nodes. 
#   - the new prediction of the tree $T_1$ is added to our initial start value and weighted by the learning parameter; now we get our prediction at iteration $1$: $f_1(x) = f_0 + \alpha T_1(\pmb{x}, y - f_0(x))$, i.e. we train a tree $T_1$ to correctly classify the residuals $y - f_0(x)$ with a rule induced on $\pmb{x}$ -- our variables. The result of this tree is weighted with leraning-rate $\alpha$ and added to the current estimate $f_0(x)$.
#   - this procedure is repeated until we can not find any more trees that substantially reduce our error or until the maximum number of iterations is reached
#   
# Since there are many different loss-functions possible for Gradient Boosting Trees (not only regression), we are looking for a more general formula that defines the best update to our current prediction made by the next tree.

# ## most important parameters for stochastic gradient-boosting:
#  - __learning_rate__: the factor $\alpha$ in the above graphic
#  - __subsample__: takes part of the data without replacement; prevents overfitting
#  - __feature_fraction__: select a subset of the features for the next tree; prevents overfitting
#  - __num_leaves__: number of leaves; lightgbm fits level-wise and leaf-wise; prevents overfitting
#  - __max_depth__: number of levels to grow the tree; prevents overfitting
#  - __num_iterations__: number of trees to grow
#  - __max_bin__: in lightgbm all features are discretized by binning them; the number of bins for a feature is given by __max_bin__ (this is what makes lightgbm super-fast)
#  - __lambda_l1/lambda_l2__: regularizes the leaf-weights; __excurs__: the result assigned to all cases that end up in one leaf is called the weight; for regression this is a continuous value and also for classification since the result is ultimately passed through a sigmoid-function that assigns then values between 0 and 1; <br>
#  
# For a speed comparison between lightgbm and xgboost see e.g. [results from 2017](https://medium.com/implodinggradients/benchmarking-lightgbm-how-fast-is-lightgbm-vs-xgboost-15d224568031) Meanwhile xgboost catched up with lightgbm.<br>
# A good paper, describing how the features of lightgbm have beed added to xgboost [is this one.](https://drive.google.com/file/d/0B0c0MbnP6Nn-eUNRRkVOOGpkbFk/view)<br>
# Less mathematical however, is this report here: 
# https://everdark.github.io/k9/notebooks/ml/gradient_boosting/gbt.nb.html#5_lightgbm
