# Additional Import Definitions
from sklearn.preprocessing import PolynomialFeatures

# Polynomial Features Initialization with 3rd Degree Polynomial
polynomial_features = PolynomialFeatures(degree=3, include_bias=False)

# Transform Feature to Polynomial
Xp = polynomial_features.fit_transform(data[['apartment_size']])
