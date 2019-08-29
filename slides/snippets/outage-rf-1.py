# Additional Import Definition
from sklearn.ensemble import RandomForestClassifier

# Hyperparameter Definitions
hyperparams = {
    'n_estimators': 4, 'criterion': 'gini', 'max_depth': None, 'min_samples_split': 2, 
    'min_samples_leaf': 1, 'min_weight_fraction_leaf': 0.0, 'max_features': 'auto', 
    'max_leaf_nodes': None, 'min_impurity_decrease': 0.0, 'min_impurity_split': None, 
    'bootstrap': True, 'oob_score': False, 'n_jobs': None, 'random_state': 1909, 
    'verbose': 0, 'warm_start': False, 'class_weight': None}

# Model Reinitialization
model = RandomForestClassifier(**hyperparams)

# Pipeline Reinitialization
pipeline = Pipeline([
    ('preprocessor', preprocessor), 
    ('model', model),
])
