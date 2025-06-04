from sklearn.base import TransformerMixin, BaseEstimator
""" BaseEstimator: provides the basic interface for all estimatprs like (.get_params() and .set_params())
TransformerMixin: adds the .fit_transform() method by combining .fit() amd .transform()"""


class MissingValueHandler(TransformerMixin, BaseEstimator):
    def __init__(self, strategy='mean', fill_value=None):
        self.strategy= strategy
        self.fill_value = fill_value
        self.fill_values_ ={}

    def fit(self, X, y=None):
        if self.strategy == 'mean':
            self.fill_values_ = X.mean().to_dict()
        elif self.strategy == 'median':
            self.fill_values_ = X.median().to_dict()
        elif self.strategy == 'mode':
            self.fill_values_ == X.mode().iloc[0].to_dict()
        elif self.strategy == 'constant':
            self.fill_values_ = {col: self.fill_value for col in X.columns}
        
        return self


    def transform(self, X):
        X=X.copy
        for col, value in self.fill_values_.items():
            X[col].fillna(value, inplace=True)
        return X
    