import pandas as pd
import numpy as np
df = pd.read_csv('single_diagnosis.csv')
print(df)
new_df = df[df["C"].str.contains("No Finding")]
print(new_df)
new_df = new_df[:500]
new_df.reset_index()

dq = df[df["C"].str.contains("Cardiomegaly")]

dq.reset_index()
dq = dq[:500]
dw = df[df["C"].str.contains("Pneumonia")]

dw.reset_index


frames = pd.concat([new_df, dq, dw])
frames.to_csv('set.csv')
