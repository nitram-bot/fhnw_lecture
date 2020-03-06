#%reload_ext autoreload
#%autoreload 2
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.cluster import KMeans
from sklearn import metrics
import nmslib
dataset = fetch_20newsgroups(subset='all', shuffle=True, download_if_missing=True)

np.random.seed(123)
texts = dataset.data # Extract text
target = dataset.target # Extract target
#texts[0:10]
#target[0:10]

vectorizer = TfidfVectorizer(stop_words='english', max_df = 0.3)
X = vectorizer.fit_transform(texts)

index = nmslib.init(method='hnsw', space='cosinesimil_sparse', data_type=nmslib.DataType.SPARSE_VECTOR)
index.addDataPointBatch(X)
index_time_params = {'post':2}
index.createIndex(index_time_params, print_progress=True)

nn = 1000
neighbors = index.knnQueryBatch(X, k=nn, num_threads=4)

col = np.array([i for n in neighbors for i in n[0].tolist()])
#row = np.repeat(np.arange(0, len(neighbors)), nn)
row = np.repeat(np.arange(0, len(neighbors)), np.array([len(n[0]) for n in neighbors]))
#data = np.array([1]*len(row))
data = np.array([i for n in neighbors for i in n[1].tolist()])
from scipy.sparse import csc_matrix
connectivity = csc_matrix((data, (row, col)), shape = (X.shape[0], X.shape[0]))
#affinity_matrix = 0.5 * (connectivity + connectivity.T)

from scipy.sparse.csgraph import laplacian as csgraph_laplacian

##### Versuch, alle eigenvalues zu bekommen #########################
from sklearn.utils import check_array
from sklearn.utils import check_random_state
from sklearn.utils.fixes import lobpcg
from sklearn.manifold._spectral_embedding import _set_diag
from scipy import sparse
from pyamg import smoothed_aggregation_solver
import matplotlib.pyplot as plt
plt.grid()

norm_laplacian = True
n_components = laplacian.shape[0]
random_state = check_random_state(1234)

laplacian, dd = csgraph_laplacian(connectivity, normed=norm_laplacian, return_diag=True)
laplacian = check_array(laplacian, dtype=np.float64, accept_sparse=True)
                                
laplacian = _set_diag(laplacian, 1, norm_laplacian)
diag_shift = 1e-5 * sparse.eye(laplacian.shape[0])
laplacian += diag_shift
ml = smoothed_aggregation_solver(check_array(laplacian, 'csr'))
laplacian -= diag_shift

M = ml.aspreconditioner()
X = random_state.rand(laplacian.shape[0], n_components + 1)
X[:, 0] = dd.ravel()
eigs, diffusion_map = lobpcg(laplacian, X, M=M, tol=1.e-5, largest=False)
plt.scatter(np.arange(len(eigs)), eigs)
plt.grid()
plt.show()
# funktioniert
#####################################################################

##
#solution = SpectralClustering(n_clusters=20, assign_labels='kmeans', \
#                              affinity='precomputed', n_neighbors=20).fit(affinity_matrix)

solution = SpectralClustering(n_clusters = 20, n_components = 21,  affinity = 'precomputed', gamma=0.7, eigen_solver='amg').fit(connectivity)
metrics.adjusted_rand_score(solution.labels_, target)

#laplacian, dd = csgraph_laplacian(affinity_matrix, normed = True, return_diag=True)
from sklearn.manifold import spectral_embedding


# this step doesn't help anything:
maps = spectral_embedding(connectivity, n_components=21, eigen_solver='amg', drop_first=False)
solutionKMeans = KMeans(n_clusters=20, init='k-means++',\
                      max_iter= 100).fit(maps)
metrics.adjusted_rand_score(solutionKMeans.labels_, target)                               
