# this fn tries to detect whether the task is classification or regression based on target variable y


from sklearn.base import is_classifier, is_regressor
import numpy as np

def detect_task_type(y):
    

    unique_values= np.unique(y)  #gets the unique values in the target
    if len(unique_values) <= 10 and np.issubdtype(y.dtype, np.integer): #checcks if dtype is sub dtype 
        return 'classification'
    else:
        return 'regression'
    

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor

def get_classification_models():
    """returns a dictionary model names to initialized classifier objects, ready to be used in
    training looops and pipelines"""
    return {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(),
        'Random Forest': RandomForestClassifier(n_estimators=100),
        'SVM': SVC(probability=True),
        'XGBoost': XGBClassifier(use_label_encoder =False, eval_metric='logloss'),
        'LightGBM': LGBMClassifier()
    }

def get_regression_models():
    """return a dictionary of commoon regression models"""
    return {
        'Linear Regression': LinearRegression(),
        'Decision Tree': DecisionTreeRegressor(),
        'Random Forest': RandomForestRegressor(n_estimators=100),
        'SVM': SVR(),
        'XGBoost': XGBRegressor(),
        'LightGBM': LGBMRegressor()
    }

