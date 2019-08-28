# Prediction Output
prediction = model.predict([[44]])
print(f'Prediction for a 44sqm Apartment: {prediction[0]:.2f} EUR')
> Prediction for a 44sqm Apartment: 482.01 EUR
    
# Parameter Output
res_params = [*model.intercept_.tolist(), *model.coef_.tolist()]
print(f'Learned Parameters: {res_params}')
> Learned Parameters: [0.24185631513843506, 10.94916974514499]

# Loss Function Calculation and Output
res_loss_function = np.sum(np.power(y - model.predict(X), 2)) * 0.5
print(f'Loss Function Value: {res_loss_function:.2f}')
> Loss Function Value: 44950.53
