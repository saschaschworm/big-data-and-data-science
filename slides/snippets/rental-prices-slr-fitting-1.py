# Import Definitions
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDRegressor

# Dataset Import
data = pd.read_csv('../datasets/demos/rental-prices.csv')

# Datset Split into Feature Matrix X and Target Vector y
X, y = data[['apartment_size']], data['rental_price']

# Hyperparameter Definitions
hyperparams = {'loss': 'squared_loss', 'penalty': 'none', 'alpha': 0.0001, 'max_iter': 1000, 
               'tol': 1e-3, 'random_state': 1909, 'eta0': 0.0001}

# Model Initialization and Fitting
model = SGDRegressor(**hyperparams)
model = model.fit(X, y)
