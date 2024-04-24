import pandas as pd

by_rows = pd.read_csv('100_cleaned.csv', header=0).to_numpy()
by_cols = by_rows.transpose()
print(by_cols[25665])
print(type(by_cols[25665][10]))
