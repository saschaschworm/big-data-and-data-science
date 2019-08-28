# Datset Split into Feature Matrix X and Target Vector y
X, y = data[['apartment_size', 'age']], data['rental_price']

# Hyperparameter Definitions
hyperparams = {'max_iter': 1000, 'tol': 1e-3, 'penalty': 'none',
                   'eta0': 0.0001, 'random_state': 1909}

# Model Initialization and Fitting
model = SGDRegressor(**hyperparams)
model = model.fit(X, y)

# Prediction Output
prediction = model.predict([[44, 10]])
print(f'Prediction for a 10-year old 44sqm Apartment: {prediction[0]:.2f} EUR')
> Prediction for a 10-year old 44sqm Apartment: 484.81 EUR
