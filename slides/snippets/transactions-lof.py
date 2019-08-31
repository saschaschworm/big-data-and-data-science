# Additional Import Definitions
from sklearn.neighbors import LocalOutlierFactor

# Data Import
data = pd.read_csv(
    'https://raw.githubusercontent.com/saschaschworm/big-data-and-data-science/'
    'master/datasets/demos/transactions.csv')

# Local Outlier Factor Initialization, Fitting and Scoring
lof = LocalOutlierFactor(n_neighbors=20, contamination='auto')
prediction = lof.fit_predict(data[['transactions', 'amount']])
scores = lof.negative_outlier_factor_

# Output Anomalies
print(data[scores < -1.1])
>                  date  transactions  amount
  date                                       
  2017-07-31 2017-07-31            24     672
  2018-06-30 2018-06-30             1    3000
