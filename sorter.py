import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

data = pd.read_csv('cardiomegaly (copy).csv')
df = pd.DataFrame(data)
print(df)

ap = 0
pa = 0
for index, row in df.iterrows():
    if 'Cardiomegaly' in row['C']
        ap += 1
    elif row['View Position'] == 'PA':
        pa += 1
print(ap)
print(pa)


objects = ('AP', 'PA')
y_pos = np.arange(len(objects))
performance = [ap, pa]

plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('View Position')
plt.title('View Positions of all X-rays')

plt.savefig('demo.pdf')
