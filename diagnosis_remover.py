import pandas as pd
import numpy as np
df = pd.read_csv('Data_Entry_2017.csv')
print(df)
new_df = df[~df["C"].str.contains("\|")]
print(new_df)
new_df.reset_index()
new_df.to_csv(r'./single_diagnosis.csv')
