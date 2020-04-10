from sklearn.linear_model import ElasticNet
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import make_scorer


class ParameterSearchNestedCV:
    """
    estimates test-performance for best-parameter search
    returns best parameters
    """
    def __init__(self, model = ElasticNet(), param_grid = [dict(alpha = 10**(np.linspace(-1,0.2,15)),
                   l1_ratio = np.linspace(0,1,10))], learning = 'regression',
                 outer_folds = 5, inner_folds = 2, n_jobs = -1 ):
        """
        model
        """
        self.model_validation = StratifiedKFold(n_splits=outer_folds,
                                                random_state=0)

        self.param_grid = param_grid

        if learning == 'regression':
            self.metric = dict(rmse = mean_squared_error,
                               r2 = r2_score)
        else:
            self.metric = dict(accuracy = accuracy_score,
                               f1 = f1_score,
                               auc = roc_auc_score)

        key = list(self.metric.keys())[0]
        self.gs = GridSearchCV(estimator = model,
                               param_grid = param_grid,
                               cv = inner_folds,
                               scoring = make_scorer(self.metric[key]),
                               n_jobs = n_jobs)
        
        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> list:
        result = []
        best_parameters = []
        for train_idx, val_idx in self.model_validation.split(X, y):
            self.gs.fit(X.loc[train_idx], y[train_idx])
            pred = self.gs.predict(X.loc[val_idx])
            result.append([self.metric[key](y_true = y[val_idx], y_pred = pred)
                for key in self.metric.keys()
                ])
            best_parameters.append(self.gs.best_params_)

        return result, best_parameters    
