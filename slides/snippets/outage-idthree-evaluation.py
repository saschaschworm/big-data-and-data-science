# Additional Import Definitions
from sklearn.metrics import make_scorer

# 4-Fold Cross-Validation with Custom F1 Scorer
custom_scorer = make_scorer(f1_score, pos_label='yes')
res_cv = cross_validate(pipeline, X, y, scoring=custom_scorer, cv=4, return_train_score=True)

# Avverage F1 Calculation and Output for Training and Test Sets
res_f1_tr = np.mean(res_cv['train_score']) * 100
res_f1_te = np.mean(res_cv['test_score']) * 100
print(f'Average F1 on Training and Test Sets: {res_f1_tr:.2f}%/{res_f1_te:.2f}%')
> Average F1 on Training and Test Sets: 100.00%/91.67%
