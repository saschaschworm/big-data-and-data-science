# List Classification Metrics
scoring = ['accuracy', 'recall', 'precision', 'f1']

# 10-Fold Cross-Validation with all Scoring Metrics
res_cv = cross_validate(pipeline, X, y, scoring=scoring, cv=10, return_train_score=True)

# Accuracy Output for Training and Test Set
res_acc_tr = np.mean(res_cv['train_accuracy']) * 100
res_acc_te = np.mean(res_cv['test_accuracy']) * 100
print(f'Average Accurarcy on Training and Test Set: {res_acc_tr:.2f}%/{res_acc_te:.2f}%')
> Average Accurarcy on Training and Test Set: 85.57%/85.45%
