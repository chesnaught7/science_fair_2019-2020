import pandas as pd
import numpy as np
df = pd.read_csv('Data_Entry_2017.csv')
print(df)
new_df = df[df["C"].str.contains("Cardiomegaly")]
length = len(new_df)
dr = df[df['C'].str.contains("No Finding")]
df = dr[:length]
dg = pd.concat([new_df, df])
print(dg)
dg.reset_index()
dg.to_csv(r'./cardiomegaly.csv')
