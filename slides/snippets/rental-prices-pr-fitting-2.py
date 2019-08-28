# Model Fitting
pipeline = pipeline.fit(X, y)

# Prediction Output
prediction = pipeline.predict(pd.DataFrame({'apartment_size': [44]}))
print(f'Prediction for a 44sqm Apartment: {prediction[0]:.2f} EUR')
> Prediction for a 44sqm Apartment: 504.18 EUR

# 10-Fold Cross-Validation with RMSE Scoring Metric
res_cv = cross_validate(pipeline, X, y, scoring=scoring, cv=10, return_train_score=True)

# RMSE Calculation and Output for Training and Test Set
res_rmse_tr = np.mean(np.sqrt(np.abs(res_cv['train_score'])))
res_rmse_te = np.mean(np.sqrt(np.abs(res_cv['test_score'])))
print(f'Average RMSE on Training and Test Set: {res_rmse_tr:.2f}/{res_rmse_te:.2f} EUR')
> Average RMSE on Training and Test Set: 42.24/49.86 EUR
