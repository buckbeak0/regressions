import pandas as pd
from sklearn.datasets import fetch_openml

boston = fetch_openml(name='boston', version=1, as_frame=True)
df = pd.DataFrame(boston.frame)

df.to_csv('Datasets/boston-housing.csv', index = False)