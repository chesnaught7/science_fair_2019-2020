import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

data = pd.read_csv('cardiomegaly.csv')
df = pd.DataFrame(data)
print(df)

ap = 0
pa = 0
for index, row in df.iterrows():
    if 'Cardiomegaly' in row['C']:
        print(row[1])
        os.rename('/media/adhit/Iomega/images/' + str(row[1]), '/media/adhit/Iomega/images/cardiomegaly/' + str(row[1]))
    elif 'No Finding' in row['C']:
        print(row[1])
        os.rename("/media/adhit/Iomega/images/" + str(row[1]), '/media/adhit/Iomega/images/nf' + str(row[1]))
