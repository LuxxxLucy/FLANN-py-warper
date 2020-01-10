"""Nearest Neighbor Classification with FLANN speedup, and in a sci-kit friendly way"""

# Authors: Jialin Lu <luxxxlucy@gmail.com>


import numpy as np
from scipy import stats
from sklearn.utils import check_array
from sklearn.neighbors.base import _check_weights, _get_weights
from pyflann import FLANN

class KNeighborsClassifier():

    def __init__(self, n_neighbors=5,weights='uniform'):
        """hyper parameters of teh FLANN algorithm"""

        self.algrithm_choice = "kmeans"
        self.branching = 32
        self.iterations = 7
        self.checks = 16

        """Basic KNN parameters"""

        self.n_neighbors = n_neighbors
        self.weights = weights
        self.flann = FLANN()





    def fit(self,X,Y):
        self.train_data = np.asarray(X).astype(np.float32)

        if Y.ndim == 1 or Y.ndim == 2 and Y.shape[1] == 1:
            if Y.ndim != 1:
                warnings.warn("A column-vector y was passed when a 1d array "
                              "was expected. Please change the shape of y to "
                              "(n_samples, ), for example using ravel().",
                              DataConversionWarning, stacklevel=2)
            print("XXXdasdasdaX!!!")
            self.outputs_2d_ = False
            Y = Y.reshape((-1, 1))
            print(Y.shape)
        else:
            self.outputs_2d_ = True

        self.classes_ = []
        self.train_label = np.empty(Y.shape, dtype=np.int)
        for k in range(self.train_label.shape[1]):
            classes, self.train_label[:, k] = np.unique(Y[:, k], return_inverse=True)
            self.classes_.append(classes)

        if not self.outputs_2d_:
            self.classes_ = self.classes_[0]
            self.train_label = self.train_label.ravel()



    def predict(self, X, n_neighbors=None):
        """Predict the class labels for the provided data.
        Parameters
        ----------
        X : array-like, shape (n_queries, n_features), \
                or (n_queries, n_indexed) if metric == 'precomputed'
            Test samples.
        Returns
        -------
        y : array of shape [n_queries] or [n_queries, n_outputs]
            Class labels for each data sample.
        """
        if n_neighbors is not None:
            self.n_neighbors = n_neighbors

        X = check_array(X, accept_sparse='csr')
        X = X.astype(np.float32)

        neigh_dist, neigh_ind = self.kneighbors(X)

        classes_ = self.classes_
        _y = self.train_label
        if not self.outputs_2d_:
            _y = self.train_label.reshape((-1, 1))
            classes_ = [self.classes_]

        n_outputs = len(classes_)
        n_queries = X.shape[0]
        weights = _get_weights(neigh_dist, self.weights)

        y_pred = np.empty((n_queries, n_outputs), dtype=classes_[0].dtype)
        for k, classes_k in enumerate(classes_):
            if weights is None:
                mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
            else:
                mode, _ = weighted_mode(_y[neigh_ind, k], weights, axis=1)

            mode = np.asarray(mode.ravel(), dtype=np.intp)
            y_pred[:, k] = classes_k.take(mode)

        if not self.outputs_2d_:
            y_pred = y_pred.ravel()

        return y_pred

        return y_pred

    def kneighbors(self,test_data):
        nearest_neighbours,dists = self.flann.nn(self.train_data,test_data,self.n_neighbors,algorithm=self.algrithm_choice, branching=self.branching, iterations=self.iterations, checks=self.checks)
        if len(nearest_neighbours.shape) == 1:
            nearest_neighbours = nearest_neighbours.reshape((-1, 1))
            dists = dists.reshape((-1, 1))
        return dists, nearest_neighbours
