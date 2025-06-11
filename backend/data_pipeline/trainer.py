

from sklearn.model_selection import cross_val_score,StratifiedKFold, KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score,f1_score, mean_squared_error
from sklearn.model_selection import GridSearchCV
import joblib
import os

class ModelTrainer:
    def __init__(self, X, y, task_type='classification', cv=5, scoring= None):
        self.X = X
        self.y = y
        self.task_type = task_type
        self.cv = cv
        self.scoring = scoring or ('accuracy' if task_type == 'classification' else 'neg_mean_squared_error')
        self.models = get_classification_models() if task_type == 'classification' else get_regression_models()

    
    def train_and_evaluate(self):
        results={}
        for name, model in self.models.items():
            try:
                print(f"Training {name}...")
                if self.task_type == 'classification':
                    cv_obj= StratifiedKFold(n_splits=self.cv)
                else:
                    cv_obj= KFold(n_splits=self.cv)
                
                scores = cross_val_score(model, self.X, self.y, cv= cv_obj, scoring= self.scoring)
                results[name] = {
                    'scores': scores,
                    'mean_score': scores.mean(),
                    'std_score': scores.std()
                    }
                print(f"{name}: Mean Score = {scores.mean():.4f}, Std = {scores.std():.4f}")
            
            except Exception as e:
                print(f"Error training {name}: {e}")
        
        return results
    

    def hyperparameter_tuning(self, model_name, param_grid, n_jobs= -1):
        model = self.models[model_name]
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv= self.cv, scoring= self.scoring, n_jobs= n_jobs)
        grid_search.fit(self.X, self.y)
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_
        
        return best_model, best_params, best_score
    

    def save_best_model(self, model, path='models/best_model.joblib'):
        os.makedirs(os.path.dirname(path), exist_ok= True)
        joblib.dump(model, path)
        print(f"Model saved at {path}")
    
        