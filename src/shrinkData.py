import pandas as pd
import gc
from importData import import_data

df = pd.read_parquet(f'../input/train.parquet')
labels = pd.read_csv(f'../input/train_labels.csv')
cid = pd.Categorical(df.pop('customer_ID'), ordered=True)

df = df.groupby(cid).mean()
df = df.head(100000)
print(df.shape)
df.to_parquet(f'../test_input/shrunk_train.parquet')

labels = labels.head(100000)
print(labels.shape)
labels.to_csv(f'../test_input/shrunk_train_label.csv')
