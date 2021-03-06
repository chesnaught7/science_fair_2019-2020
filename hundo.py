import pandas as pd
import numpy as np
df = pd.read_csv('single_diagnosis.csv')
print(df)
new_df = df[df["C"].str.contains("Cardiomegaly")]
new_da = new_df[:300]

da = df[df['C'].str.contains("No Finding")]
db = da[:300]
dg = pd.concat([new_da, db])
print(dg)
dg.reset_index()
dg.to_csv(r'./quintihundo.csv')
