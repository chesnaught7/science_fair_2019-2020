import pandas as pd
import numpy as np
df = pd.read_csv('single_diagnosis.csv')
print(df)
new_df = df[df["C"].str.contains("Cardiomegaly")]
new_df = new_df[:100]
da = df[df['C'].str.contains("No Finding")]
da = da[:100]
dg = pd.concat([new_df, da])
print(dg)
dg.reset_index()
dg.to_csv(r'./hundo.csv')
