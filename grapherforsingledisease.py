import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
df = pd.read_csv('single_diagnosis.csv')
print(df)
a = 0
b = 0
c = 0
d = 0
e = 0
f = 0
g = 0
h = 0
i = 0
j = 0
k = 0
l = 0
m = 0
n = 0
o = 0
new_df = df[df["C"].str.contains("Atelectasis")]
df_2 = df[df["C"].str.contains("Nodule")]
df_3 = df[df["C"].str.contains("Emphysema")]
df_4 = df[df["C"].str.contains("Pneumonia")]
df_5 = df[df["C"].str.contains("Fibrosis")]
df_6 = df[df["C"].str.contains("Consolidation")]
df_7 = df[df["C"].str.contains("Cardiomegaly")]
df_8 = df[df["C"].str.contains("Pleural_Thickening")]
df_9 = df[df["C"].str.contains("Effusion")]
df_10 = df[df["C"].str.contains("Infiltration")]
df_11 = df[df["C"].str.contains("Hernia")]
df_12 = df[df["C"].str.contains("Pneumothorax")]
df_13 = df[df["C"].str.contains("Mass")]
df_15 = df[df["C"].str.contains("Edema")]

df_4.to_csv("hehd.csv")
for index, row in df_2.iterrows():
    b += 1
for index, row in new_df.iterrows():
    a += 1
for index, row in df_3.iterrows():
    c += 1

for index, row in df_4.iterrows():
    d += 1
for index, row in df_5.iterrows():
    e += 1
for index, row in df_6.iterrows():
    f+= 1
for index, row in df_7.iterrows():
    g+= 1
for index, row in df_8.iterrows():
    h+= 1

for index, row in df_9.iterrows():
    i+= 1
for index, row in df_10.iterrows():
    j+= 1
for index, row in df_11.iterrows():
    k+= 1

for index, row in df_12.iterrows():
    l += 1
for index, row in df_13.iterrows():
    m += 1
for index, row in df_15.iterrows():
    o += 1

objects = ('Atelectasis', 'Nodule', "Emphysema", "Pneumonia", "Fibrosis", "Consolidation", "Cardiomegaly", "Pleural Thickening", "Effusion", "Infiltration", "Hernia", "Pneumothorax", "Mass", "Edema")
y_pos = np.arange(len(objects))
performance = [a,b, c, d,e,f,g,h,i,j,k,l,m,o]

font = {'family' : 'normal','weight' : 'bold','size'   : 8}
fig, ax = plt.subplots()

ax.barh(y_pos, performance, align='center')
ax.set_yticks(y_pos)
ax.set_yticklabels(objects)
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Number of Instances')
ax.set_title("Disease Distribution")
ax.tick_params(axis='both', which='major', labelsize=5)
ax.tick_params(axis='both', which='minor', labelsize=5)


plt.savefig('single_disease_distribution_graph.png')
plt.show()
