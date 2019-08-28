# Additional Import Definitions
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

numeric_features = ['apartment_size']
numeric_transformer = Pipeline([
    ('polynomials', PolynomialFeatures(degree=3, include_bias=False)),
    ('scaler', MinMaxScaler()),
])

preprocessor = ColumnTransformer([
    ('numeric_transformer', numeric_transformer, numeric_features),
])

pipeline = Pipeline([
    ('preprocessor', preprocessor), 
    ('model', SGDRegressor(max_iter=50000, penalty='none', eta0=0.01, random_state=1909))
])
