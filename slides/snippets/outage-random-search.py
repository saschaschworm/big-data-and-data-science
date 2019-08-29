# Additional Import Definition
from scipy.stats import randint
from sklearn.model_selection import RandomizedSearchCV

# Hyperparameter Uniform Sampling Distribution
n_estimators = randint(100, 500)
max_depth = randint(1, 10)
param_distributions = {'model__n_estimators': n_estimators, 'model__max_depth': max_depth}

# Initialize Randomized Search
rs = RandomizedSearchCV(pipeline, param_distributions=param_distributions, n_iter=5, 
                        scoring=custom_scorer, n_jobs=-1, iid=False, cv=4, random_state=1909)

# Begin Randomized Search with 4-fold Cross Validation
rs = rs.fit(X, y)

# Output Best Parameters
print(f'Optimal parameters: {rs.best_params_}')
> Optimal parameters: {'model__max_depth': 6, 'model__n_estimators': 442}
