# Additional Import Definitions
from sklearn.metrics import mean_squared_error

# RMSE Calculation and Output
res_rmse = np.sqrt(mean_squared_error(y, model.predict(X)))
print(f'RMSE on the Training Set: {res_rmse:.2f} EUR')
> RMSE on the Training Set: 47.41 EUR
