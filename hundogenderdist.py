import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
m = 0
f = 0

data = pd.read_csv('./cardiomegaly.csv')
df = pd.DataFrame(data)
print(df)
new_df = df[df["Patient Gender"].str.contains("m")]
df_1 = df[df["Patient Gender"].str.contains("f")]

for index, row in new_df.iterrows():
    m += 1
for index, row in df_1.iterrows():
    f += 1
objects = ("Male", "Female")
y_pos = np.arange(len(objects))
performance = [m, f]

font = {'family' : 'normal','weight' : 'bold','size'   : 8}
fig, ax = plt.subplots()

ax.barh(y_pos, performance, align='center')
ax.set_yticks(y_pos)
ax.set_yticklabels(objects)
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Gender')
ax.set_title("Gender Distribution for Cardiomegaly")
ax.tick_params(axis='both', which='major', labelsize=5)
ax.tick_params(axis='both', which='minor', labelsize=5)


plt.savefig('genderdistnf.pdf')
plt.show()
