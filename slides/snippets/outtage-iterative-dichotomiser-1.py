# Additional Python Packages
from sklearn.tree import DecisionTreeClassifier

# Dataset Import
data = pd.read_csv(
    'https://raw.githubusercontent.com/saschaschworm/big-data-and-data-science/'
    'master/datasets/demos/outages.csv')

# Datset Split into Feature Matrix X and Target Vector y
X, y = data.iloc[:, 0:3], data['outage']

# Hyperparameter Definitions
hyperparams = {'criterion': 'entropy', 'max_depth': None, 'min_samples_split': 2, 
               'min_samples_leaf': 1, 'min_weight_fraction_leaf': 0.0, 'max_features': None, 
               'random_state': 1909, 'max_leaf_nodes': None, 'min_impurity_decrease':0.0, 
               'min_impurity_split': None}

# Model Initialization
model = DecisionTreeClassifier(**hyperparams)
