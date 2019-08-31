# Additional Import Definitions
from statsmodels.tsa.seasonal import seasonal_decompose

# Data Import
data = pd.read_csv(
    'https://raw.githubusercontent.com/saschaschworm/big-data-and-data-science/'
    'master/datasets/demos/air-passengers.csv')

# Feature and Index Transformation
data['date'] = pd.to_datetime(data['date'], format='%Y-%m-%d')
data.index = data['date']

# Multiplicative Seasonal and Trend Decomposition
stl = seasonal_decompose(data['passengers'], 'multiplicative')
