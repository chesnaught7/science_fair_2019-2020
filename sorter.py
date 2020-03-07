import matplotlib
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

data = pd.read_csv('set.csv')
df = pd.DataFrame(data)
print(df)

ap = 0
pa = 0
a = 0
b = 0
c = 0
for index, row in df.iterrows():
    if 'Cardiomegaly' in row['C']:
        print(row['C'])
        os.rename('/media/adhit/Iomega/images/' + str(row['Image Index']), '/media/adhit/Iomega/images/cardiomegaly/' + str(row['Image Index']))


    elif 'No Finding' in row['C']:
        print(row['C'])
        os.rename("/media/adhit/Iomega/images/" + str(row['Image Index']), '/media/adhit/Iomega/images/nf/' + str(row['Image Index']))


    elif 'Pneumonia' in row['C']:
            print(row['C'])
            os.rename("/media/adhit/Iomega/images/" + str(row['Image Index']), '/media/adhit/Iomega/images/pneumonia/' + str(row['Image Index']))
