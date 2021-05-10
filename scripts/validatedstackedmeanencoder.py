import pandas as pd
import numpy as np
from category_encoders.james_stein import JamesSteinEncoder
from category_encoders.cat_boost import CatBoostEncoder
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin



class ValidatedStackedMeanEncoder(BaseEstimator, TransformerMixin):
    """
    repeated cross-validation in constructing mean-encoding
    """
    def __init__(self):
        """
        :param cols: Categorical columns
        :param encoders_names_tuple: Tuple of str with encoders
        """
        #target_encoder = JamesSteinEncoder
        self.target_encoder = CatBoostEncoder
        k_folds = 5
        n_repeats = 3
        
        self.k_folds, self.n_repeats = k_folds, n_repeats
        self.model_validation = RepeatedStratifiedKFold(n_splits=self.k_folds, n_repeats=self.n_repeats, random_state=0)
        self.list_of_encoders = []
        self._fit_index = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        X = pd.DataFrame(X).reset_index(drop=True)
        y = pd.Series(y).reset_index(drop=True)
        self._fit_index = X.index
        self.list_of_encoders=[]

        for n_fold, (train_idx, val_idx) in enumerate(self.model_validation.split(X, y)):
            encoder = self.target_encoder(cols=X.columns)
            X_train = X.loc[train_idx, :]
            y_train = y.loc[train_idx]
            encoder.fit(X_train, y_train)
            self.list_of_encoders.append(encoder)

        return self

    def fit_transform(self, X, y):
        X = pd.DataFrame(X).reset_index(drop=True)
        y = pd.Series(y).reset_index(drop=True)
        cols_representation = np.zeros_like(X, dtype=float)
        cols = X.columns
        self.list_of_encoders = []
        for n_fold, (train_idx, val_idx) in enumerate(self.model_validation.split(X, y)):
            print('in fit_transform')
            
            encoder = self.target_encoder(cols=X.columns)
            X_train, X_val = X.loc[train_idx, :], X.loc[val_idx, :]
            y_train, y_val = y[train_idx], y[val_idx]
            encoder.fit(X_train, y_train)

            # transform validation part and get all necessary cols
            val_t = encoder.transform(X_val)

            cols_representation[val_idx, :] += val_t / self.n_repeats
            self.list_of_encoders.append(encoder)

        cols_representation = pd.DataFrame(cols_representation, columns=['mean_enc_' + c for c in cols])
        X = pd.concat([X, cols_representation], axis=1)
        return X

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = pd.DataFrame(X).reset_index(drop=True)
        cols_representation = np.zeros_like(X, dtype=float)
        cols = X.columns
        if self._fit_index and len(self._fit_index) == len(X.index):
            print('in if of transform')
            for n_fold, (train_idx, val_idx) in enumerate(self.model_validation.split(X)):
                val_t = self.list_of_encoders[n_fold].transform(X.loc[val_idx].reset_index(drop=True))
                cols_representation[val_idx, :] += val_t / self.n_repeats
        else:
            print('in else of transform')

            for encoder in self.list_of_encoders:
                test_tr = encoder.transform(X)
                cols_representation += test_tr / (self.k_folds * self.n_repeats)

        cols_representation = pd.DataFrame(cols_representation.values, columns=['mean_enc_' + c for c in cols])
        X = pd.concat([X, cols_representation], axis=1)
        # X.drop(cols, axis=1, inplace=True)
        return X
