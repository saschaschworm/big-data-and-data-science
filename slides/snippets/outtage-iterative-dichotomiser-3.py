# Prediction Dataset Initialization
instance = pd.DataFrame({'vibration': ['medium'], 'error': ['yes'], 'temperature': ['low']})

# Prediction and Output
prediction = pipeline.predict(instance)
prediction_proba = pipeline.predict_proba(instance)
print(f'Prediction result: {prediction} ({prediction_proba})')
> Prediction result: ['no'] ([[1. 0.]]
