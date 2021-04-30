import pandas as pd
from category_encoders.james_stein import JamesSteinEncoder
from category_encoders.cat_boost import CatBoostEncoder
from sklearn.model_selection import RepeatedStratifiedKFold


class ValidatedStackedMeanEncoder:
    """
    repeated cross-validation in constructing mean-encoding
    """
    def __init__(self, cols: list, target_encoder=JamesSteinEncoder, k_folds: int = 5, repeats: int = 3):
        """
        :param cols: Categorical columns
        :param encoders_names_tuple: Tuple of str with encoders
        """
        self.cols, self.num_cols = cols, None
        self.encoder = target_encoder
        
        self.n_folds, self.n_repeats = k_folds, repeats
        self.model_validation = RepeatedStratifiedKFold(n_splits=self.n_folds, n_repeats=self.n_repeats, random_state=0)
        self.list_of_encoders = []

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        self.num_cols = [col for col in X.columns if col not in self.cols]

        cols_representation = np.zeros((X.shape[0], len(self.cols)))

        for n_fold, (train_idx, val_idx) in enumerate(self.model_validation.split(X, y)):
            
            encoder = self.encoder(self.cols)
            X_train, X_val = X.loc[train_idx].reset_index(drop=True), X.loc[val_idx].reset_index(drop=True)
            y_train, y_val = y[train_idx], y[val_idx]
            _ = encoder.fit_transform(X_train, y_train)

            # transform validation part and get all necessary cols
            val_t = encoder.transform(X_val)
            val_t = val_t[[col for col in val_t.columns if col not in self.num_cols]].values

            cols_representation[val_idx, :] += val_t / self.n_repeats
            self.list_of_encoders.append(encoder)

        X = pd.concat([X, pd.DataFrame(cols_representation)], axis = 1)
        X.drop(self.cols, axis=1, inplace=True)
        return X

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        cols_representation = np.zeros((X.shape[0], len(self.cols)))

        for encoder in self.list_of_encoders:
            test_tr = encoder.transform(X)
            test_tr = test_tr[[col for col in test_tr.columns if col not in self.num_cols]].values

            cols_representation = cols_representation + test_tr / self.n_folds / self.n_repeats

        cols_representation = pd.DataFrame(cols_representation)
        X = pd.concat([X, cols_representation], axis=1)
        X.drop(self.cols, axis=1, inplace=True)
        return X
