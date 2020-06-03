import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
m = []
mu = 100  # mean of distribution
sigma = 15  # standard deviation of distribution
data = pd.read_csv('single_diagnosis.csv')
df = pd.DataFrame(data)
print(df)


for index, row in df.iterrows():
    m.append(row['Patient Age'])
objects = ("Male", "Female")
fig, ax = plt.subplots()

# the histogram of the data
n, bins, patches = ax.hist(m, bins = 80)

# add a 'best fit' line

ax.set_xlabel('Age')
ax.set_ylabel('Instances of Age')
ax.set_title('Age Distribution')

# Tweak spacing to prevent clipping of ylabel
fig.tight_layout()
plt.xlim(right=100)
plt.xlim(left = 0)
plt.savefig('age_dist.pdf')
plt.show()
