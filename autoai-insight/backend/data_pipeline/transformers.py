#===================================
#Handle Missing Values:
#===================================

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
    

#=====================================
#Encode categorical features:
#=====================================

from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
"""OneHotEncoder: converts categorical variable into binary vectors(one-hot)
OrdinalEncoder: maps categories to integer values(0,1,2,.....)
#TargetEncoder: encodes categories based on target mean(supervised)"""
from category_encoders import TargetEncoder

class CategoricalEncoder(TransformerMixin, BaseEstimator):
    def __init__(self, encoding_method='onehot', target_col=None):
        """sets up the encoding strategy
        stores learned encoders per column in self.encoders_"""
        self.encoding_method = encoding_method
        self.target_col = target_col
        self.encoders_ = {}

    def fit(self, X, y=None):
        for col in X.columns:
            if self.encoding_method == 'onehot':
                encoder = OneHotEncoder(handle_unknown='ignore', sparse_output = False)
                encoder.fit(X[[col]])
            elif self.encoding_method == 'ordinal':
                encoder = OrdinalEncoder()
                encoder.fit(X[[col]])
            elif self.encoding_method == 'target' and y is not None:
                encoder = TargetEncoder()
                encoder.fit(X[col],y)
            else:
                raise ValueError("Unsupported encoding method or missing target for target encoding.")
            self.encoders_[col] = encoder
        return self
    

    def transform(self, X):
        X = X.copy()
        for col, encoder in self.encoders_.items():
            X[col] = encoder.transform(X[[col]])
        return X
    

#======================================
#Scale Numerical Features:
#======================================

from sklearn.preprocessing import StandardScaler, MinMaxScaler
""" StandardScaler():  sclers data to have zero mean and unit variance(standard normal distribution)
MinMaxScaler: scalers data to a fixed range (default[0,1])
Crucial for models sensetive to features like KNN, SVM, LogReg, NN
"""

class FeatureScaler(TransformerMixin, BaseEstimator):
    def __init__(self, scaling_method='standard'):
        self.scaling_method = scaling_method
        self.scalers_ = {}

    def fit(self, X, y=None):
        for col in X.columns:
            if self.scaling_method == 'standard':
                scaler = StandardScaler()
            elif self.scaling_method == 'minmax':
                scaler = MinMaxScaler()
            else:
                raise ValueError("Unsupported scaling method")
            
            scaler.fit(X[[col]])
            self.scalers_[col] = scaler

        return self
    

    def transform(self, X):
        X= X.copy
        for col, scaler in self.scalers_.items():
            X[col] = scaler.transform(X[[col]])
        return X
    



