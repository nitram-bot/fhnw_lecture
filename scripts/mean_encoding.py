# box cox transform
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from random import choices
import category_encoders as ce
from category_encoders.james_stein import JamesSteinEncoder
from category_encoders.cat_boost import CatBoostEncoder
from sklearn.model_selection import RepeatedStratifiedKFold


# read in data
train = pd.read_csv('../data/train.csv')
test = pd.read_csv('../data/test.csv')

train_ID = train['Id']
test_ID = test['Id']

train.drop('Id', axis = 1, inplace = True)
test.drop('Id', axis = 1, inplace = True)

SalePrice = train['SalePrice']
train.drop('SalePrice', axis=1, inplace = True)

categorical = [var for var in train.columns if train[var].dtype=='O']
numerical = [var for var in train.columns if train[var].dtype!='O']

train[categorical] = train[categorical].fillna('None')


# JamesSteinEncoder
# CatBoostEncoder

encJS = JamesSteinEncoder(cols = categorical)
encCB = CatBoostEncoder(cols = categorical)

class DoubleValidationEncoderNumerical:
    """
    Encoder with validation within
    """
    def __init__(self, cols: list, target_encoder: category_encoders = JamesSteinEncoder, k-folds: int = 5, repeats: int = 3):
        """
        :param cols: Categorical columns
        :param encoders_names_tuple: Tuple of str with encoders
        """
        self.cols, self.num_cols = cols, None
        self.encoder = target_encoder(self.cols)
        
        self.n_folds, self.n_repeats = k-folds, repeats
        self.model_validation = RepeatedStratifiedKFold(n_splits=self.n_folds, n_repeats=self.n_repeats, random_state=0)
        self.list_of_encoders = []

        self.storage = None

    def fit_transform(self, X: pd.DataFrame, y: np.array) -> pd.DataFrame:
        self.num_cols = [col for col in X.columns if col not in self.cols]

        cols_representation = np.zeros((X.shape[0], val_t.shape[1]))

        for n_fold, (train_idx, val_idx) in enumerate(self.model_validation.split(X, y)):
 
            X_train, X_val = X.loc[train_idx].reset_index(drop=True), X.loc[val_idx].reset_index(drop=True)
            y_train, y_val = y[train_idx], y[val_idx]
            _ = self.encoder.fit_transform(X_train, y_train)

            # transform validation part and get all necessary cols
            val_t = encoder.transform(X_val)
            val_t = val_t[[col for col in val_t.columns if col not in self.num_cols]].values

            cols_representation[val_idx, :] += val_t / self.n_repeats
            self.list_of_encoders.append(encoder)

    X = pd.concat([X, pd.DataFrame(cols_representation)], axis = 1)
    X.drop(self.cols, axis=1, inplace=True)
    return X



    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self.storage = []
        for encoder_name in self.encoders_names_tuple:
            cols_representation = None

            for encoder in self.encoders_dict[encoder_name]:
                test_tr = encoder.transform(X)
                test_tr = test_tr[[col for col in test_tr.columns if col not in self.num_cols]].values

                if cols_representation is None:
                    cols_representation = np.zeros(test_tr.shape)

                cols_representation = cols_representation + test_tr / self.n_folds / self.n_repeats

            cols_representation = pd.DataFrame(cols_representation)
            cols_representation.columns = [f"encoded_{encoder_name}_{i}" for i in range(cols_representation.shape[1])]
            self.storage.append(cols_representation)

        for df in self.storage:
            X = pd.concat([X, df], axis=1)

        X.drop(self.cols, axis=1, inplace=True)
        return X

