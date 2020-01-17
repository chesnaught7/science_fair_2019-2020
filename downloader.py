from google.cloud import storage
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib

df = pd.read_csv('hehedbigbig.csv')
bucket_name = "lung_x-ray"
source_blob_name = ""
destination_file_name = "/media/adhit/OS/Users/adhit/Lung X-rays/"
storage_client = storage.Client()
bucket = storage_client.bucket(bucket_name)
a = 1
for index, row in df.iterrows():
    print(row)
    source_blob_name = row[1]
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name + str(a))
    a += 1
print("Blob {} downloaded to {}.".format(source_blob_name, destination_file_name))
