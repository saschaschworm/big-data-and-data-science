# Additional Import Definitions
from sklearn.model_selection import train_test_split

# Feature Matrix and Target Vector Split into Training and Test Set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1909)

# Model Refitting
model = model.fit(X_train, y_train)

# Prediction for Training and Test Set Features
y_pred_tr, y_pred_te = model.predict(X_train), model.predict(X_test) 

# RMSE Calculation and Output for Training and Test Set
res_rmse_tr = np.sqrt(mean_squared_error(y_train, y_pred_tr))
res_rmse_te = np.sqrt(mean_squared_error(y_test, y_pred_te))
print(f'RMSE on Training and Test Set: {res_rmse_tr:.2f}/{res_rmse_te:.2f} EUR')
> RMSE on Training and Test Set: 50.58/36.41 EUR
