# Model Reinitialization
model = RandomForestClassifier(n_estimators=442, max_depth=6, random_state=1909)

# Pipeline Reinitialization and Fitting
pipeline = Pipeline([
    ('preprocessor', preprocessor), 
    ('model', model),
])

pipeline = pipeline.fit(X, y)

# Average F1 Calculation and Output for Training and Test Sets
res_cv = cross_validate(pipeline, X, y, scoring=custom_scorer, cv=4, return_train_score=True)
res_f1_tr = np.mean(res_cv['train_score']) * 100
res_f1_te = np.mean(res_cv['test_score']) * 100
print(f'Average F1 on Training and Test Sets: {res_f1_tr:.2f}%/{res_f1_te:.2f}%')
> Average F1 on Training and Test Sets: 100.00%/96.43%
