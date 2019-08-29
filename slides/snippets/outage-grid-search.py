# Additional Import Definition
from sklearn.model_selection import GridSearchCV

# Hyperparameter Grid Setup
n_estimators = [100, 200, 300, 400, 500]
max_depth = [1, 2, 3, 5, 6, 7, 8, 9, 10]
param_grid = {'model__n_estimators': n_estimators,  'model__max_depth': max_depth}

# Initialize Grid Search
gs = GridSearchCV(pipeline, param_grid=param_grid, scoring=custom_scorer, n_jobs=-1, 
                  iid=False, cv=4)
gs = gs.fit(X, y)

# Begin Grid Search with 4-fold Cross Validation
gs = grid_search_cv.fit(X, y)

# Output Best Parameters
print(f'Optimal parameters: {gs.best_params_}')
> Optimal parameters: {'model__max_depth': 3, 'model__n_estimators': 400}
