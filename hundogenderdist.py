import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
data = pd.read_csv('./hundo.csv')
df = pd.DataFrame(data)
print(df)
m = 0
f = 0
df = df[~df["C"].str.contains("No Finding")]
new_df = df[~df["Patient Gender"].str.contains("M")]
dr = df[~df["Patient Gender"].str.contains("F")]


for index, row in new_df.iterrows():
    m += 1
for index, row in dr.iterrows():
    f += 1

objects = ("M", "F")
y_pos = np.arange(len(objects))
performance = [m , f]

font = {'family' : 'normal','weight' : 'bold','size'   : 8}
fig, ax = plt.subplots()

ax.barh(y_pos, performance, align='center')
ax.set_yticks(y_pos)
ax.set_yticklabels(objects)
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Gender')
ax.set_title("Gender Distribution")
ax.tick_params(axis='both', which='major', labelsize=5)
ax.tick_params(axis='both', which='minor', labelsize=5)

plt.show()
plt.savefig('hundogenderdistnf.pdf')
