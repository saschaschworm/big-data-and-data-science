# Data Import
data = pd.read_csv(
    'https://raw.githubusercontent.com/saschaschworm/big-data-and-data-science/'
    'master/datasets/demos/taxi-passengers.csv')

# Feature and Index Transformation
data['date'] = pd.to_datetime(data['date'], format='%Y-%m-%d')
data.index = data['date']

# 20-day Moving Average Calculation
data['mean20d'] = data['passengers'].rolling(20).mean()

# 20-day Moving Standard Deviation Calculation
data['std20dmean'] = data['passengers'].rolling(20).std()
