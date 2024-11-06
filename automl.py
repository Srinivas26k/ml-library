# automl.py
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

class AutoML:
    def __init__(self, models=None, param_grids=None):
        # Define models and their hyperparameter grids
        self.models = models or {
            'Random Forest': RandomForestClassifier(),
            'Logistic Regression': LogisticRegression(),
            'SVM': SVC()
        }
        self.param_grids = param_grids or {
            'Random Forest': {'n_estimators': [10, 50, 100]},
            'Logistic Regression': {'C': [0.1, 1, 10]},
            'SVM': {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
        }

    def train_and_tune(self, X, y):
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        best_model = None
        best_score = 0
        
        for model_name, model in self.models.items():
            grid_search = GridSearchCV(model, self.param_grids[model_name], cv=3)
            grid_search.fit(X_train, y_train)
            score = grid_search.best_score_
            
            if score > best_score:
                best_score = score
                best_model = grid_search.best_estimator_
        
        return best_model, best_score
