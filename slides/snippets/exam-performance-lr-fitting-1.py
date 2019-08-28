# Additional Import Definitions
from sklearn.linear_model import SGDClassifier

# Dataset Import
data = pd.read_csv('../datasets/demos/exam-performance.csv')

# Datset Split into Feature Matrix X and Target Vector y
X, y = data[['hours_studied', 'hours_slept']], data['passed']

# Hyperparameter Definitions
hyperparams = {'loss': 'log', 'penalty': 'none', 'alpha': 0.0001, 'max_iter': 1000, 
               'tol': 1e-3, 'random_state': 1909, 'eta0': 0.0001}

# Model Initialization and Fitting
model = SGDClassifier(**hyperparams)
model = model.fit(X, y)
