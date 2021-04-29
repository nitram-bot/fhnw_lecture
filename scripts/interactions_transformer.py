
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from itertools import combinations
import pandas as pd


class InteractionsTransformer(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.gbm = GradientBoostingRegressor(n_estimators=32, max_depth=4)
        self.sorted_idx = None

    def fit(self, X: pd.DataFrame, y=None):
        self.gbm.fit(X, y)
        result = permutation_importance(self. gbm, X, y, n_repeats=10,
                               random_state=42, n_jobs=2)
        self.sorted_idx = result.importances_mean.argsort()

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        for comb in list(combinations(X.columns[self.sorted_idx[-15:]], 2)):
            X[comb[0] + '_x_' + comb[1]] = X[comb[0]] * X[comb[1]]

        return X

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y)
        out = self.transform(X)

        return out
