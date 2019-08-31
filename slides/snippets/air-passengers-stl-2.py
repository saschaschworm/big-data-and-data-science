# Lower and Upper Quantile Calculation
upper_threshold = np.quantile(stl.resid.dropna(), 0.99)
lower_threshold = np.quantile(stl.resid.dropna(), 0.01)

# Anomaly Detection
anomalies = data.loc[
    (stl.resid > upper_threshold) | 
    (stl.resid < lower_threshold), :]

# Output Anomalies
print(anomalies['passengers'])
> date
  1952-02-01    180
  1952-04-01    150
  1953-04-01    235
  1960-03-01    419
  Name: passengers, dtype: int64
