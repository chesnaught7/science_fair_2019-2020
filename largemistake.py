import os

path = '/media/adhit/Iomega/images/'

files = []
# r=root, d=directories, f = files
for r, d, f in os.walk(path):
    for file in f:
        if 'nf' in file:
            files.append(os.path.join(r, file))

for f in files:
    g = f.replace('//', '/')
    os.rename(f, g)
