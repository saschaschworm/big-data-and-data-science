# Preprocessing
categorical_featues = ['vibration', 'error', 'temperature']

categorical_transformer = Pipeline([
    ('onehotencoder', OneHotEncoder(drop='first')),
])

preprocessor = ColumnTransformer([
    ('categorical_transformer', categorical_transformer, categorical_featues),
])

pipeline = Pipeline([
    ('preprocessor', preprocessor), 
    ('model', model),
])

# Pipeline and Model Fitting
pipeline.fit(X, y)
