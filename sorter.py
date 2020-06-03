import matplotlib
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

data = pd.read_csv('single_diagnosis.csv')
df = pd.DataFrame(data)
print(df)

ap = 0
pa = 0
a = 0
b = 0
c = 0
for index, row in df.iterrows():
    if 'Cardiomegaly' in row['C']:
        print(row['Image Index'])
        os.rename('../Downloads/images/' + str(row['Image Index']), '../Downloads/images/cardiomegaly/' + str(row['Image Index']))


    elif 'No Finding' in row['C']:
        print(row['Image Index'])
        os.rename("../Downloads/images/" + str(row['Image Index']), '../Downloads/images/nf/' + str(row['Image Index']))


    elif 'Pneumonia' in row['C']:
            print(row['C'])
            os.rename("../Downloads/images/" + str(row['Image Index']), '../Downloads/images/pneumonia/' + str(row['Image Index']))
    elif 'Atelectasis' in row['C']:
        print(row['Image Index'])
        os.rename("../Downloads/images/" + str(row['Image Index']), '../Downloads/images/atelectasis/' + str(row['Image Index']))
    elif 'Effusion' in row['C']:
        print(row['Image Index'])
        os.rename("../Downloads/images/" + str(row['Image Index']), '../Downloads/images/effusion/' + str(row['Image Index']))
    elif 'Pneumothorax' in row['C']:
        print(row['Image Index'])
        os.rename("../Downloads/images/" + str(row['Image Index']), '../Downloads/images/pneumothorax/' + str(row['Image Index']))
    elif 'Infiltration' in row['C']:
        print(row['Image Index'])
        os.rename("../Downloads/images/" + str(row['Image Index']), '../Downloads/images/infiltration/' + str(row['Image Index']))
    elif 'Mass' in row['C']:
        print(row['Image Index'])
        os.rename("../Downloads/images/" + str(row['Image Index']), '../Downloads/images/mass/' + str(row['Image Index']))
    elif 'Nodule' in row['C']:
        print(row['Image Index'])
        os.rename("../Downloads/images/" + str(row['Image Index']), '../Downloads/images/nodule/' + str(row['Image Index']))
