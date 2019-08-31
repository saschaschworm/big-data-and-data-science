# Threshold Calculation with Three-Sigma Rule
upper_threshold = data['mean20d'] + 3 * data['std20dmean']
lower_threshold = data['mean20d'] - 3 * data['std20dmean']

# Anomaly Detection
anomalies = data[(data['passengers'] > upper_threshold) | 
                 (data['passengers'] < lower_threshold)]

# Output Anomalies
print(anomalies['passengers'])
> date
  2014-12-25    379302
  2015-01-26    375311
  2015-01-27    232058
  Name: passengers, dtype: int64
