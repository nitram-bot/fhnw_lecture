import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.metrics import accuracy_score, auc, log_loss

class MyCustomEstimator(BaseEstimator, ClassifierMixin):

    def __init__(self, parameters)
        self.parameters = 0

    def fit(self, X, y):

        return self

    def predict(self, X):
        return y_pred

    def get_params(self):
        """
        returns dictionary with parameters
        """
        return self.parameters

    def set_params(self, **param_dict):
        """
        alternatively new value in self.param_dict
        """
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
        
