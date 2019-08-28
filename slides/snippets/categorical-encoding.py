# Additional Import Definitions
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Label Encoder Pipeline Initialization
categorical_transformer = Pipeline([
    ('onehotencoder', LabelEncoder()), #or
    ('labelencoder', OneHotEncoder(drop='first')),
])
