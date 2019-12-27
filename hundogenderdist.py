import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


data = pd.read_csv('./hundo.csv')
df = pd.DataFrame(data)
print(df)

df.groupby("Patient Gender").hist()
