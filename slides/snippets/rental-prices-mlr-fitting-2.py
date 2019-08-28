# 10-Fold Cross-Validation with RMSE Scoring Metric
res_cv = cross_validate(model, X, y, scoring=scoring, cv=10, return_train_score=True)

# RMSE Calculation and Output for Training and Test Set
res_rmse_tr = np.mean(np.sqrt(np.abs(res_cv['train_score'])))
res_rmse_te = np.mean(np.sqrt(np.abs(res_cv['test_score'])))
print(f'Average RMSE on Training and Test Set: {res_rmse_tr:.2f} EUR/{res_rmse_te:.2f} EUR')
> Average RMSE on Training and Test Set: 39.03 EUR/38.13 EUR
