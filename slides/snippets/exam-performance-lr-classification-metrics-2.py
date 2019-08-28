# Recall Output for Training and Test Set
res_rec_tr = np.mean(res_cv['train_recall']) * 100
res_rec_te = np.mean(res_cv['test_recall']) * 100
print(f'Average Recall on Training and Test Set: {res_rec_tr:.2f}%/{res_rec_te:.2f}%')
> Average Recall on Training and Test Set: 92.93%/90.33%

# Precision Output for Training and Test Set
res_prec_tr = np.mean(res_cv['train_precision']) * 100
res_prec_te = np.mean(res_cv['test_precision']) * 100
print(f'Average Precision on Training and Test Set: {res_prec_tr:.2f}%/{res_prec_te:.2f}%')
> Average Precision on Training and Test Set: 84.16%/87.62%

# F1 Output for Training and Test Set
res_f1_tr = np.mean(res_cv['train_f1']) * 100
res_f1_te = np.mean(res_cv['test_f1']) * 100
print(f'Average F1 on Training and Test Set: {res_f1_tr:.2f}%/{res_f1_te:.2f}%')
> Average F1 on Training and Test Set: 87.67%/86.59%
