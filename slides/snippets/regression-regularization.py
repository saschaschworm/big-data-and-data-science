# Model Initialization with LASSO (L1)
hyperparams = {'loss': 'squared_loss', 'penalty': 'l1', 'alpha': 0.0001, 'max_iter': 1000, 
               'tol': 1e-3, 'random_state': 1909, 'eta0': 0.0001}
model = SGDRegressor(**hyperparams)

# Model Initialization with RIDGE (L2)
hyperparams = {'loss': 'squared_loss', 'penalty': 'l2', 'alpha': 0.0001, 'max_iter': 1000, 
               'tol': 1e-3, 'random_state': 1909, 'eta0': 0.0001}
model = SGDRegressor(**hyperparams)
                     
# Model Fitting
model = model.fit(X, y)
