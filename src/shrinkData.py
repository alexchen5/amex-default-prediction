import pandas as pd
import gc
from importData import import_data

# Shrink data to specified size for faster computations

df = pd.read_parquet(f'../input/train.parquet')
labels = pd.read_csv(f'../input/train_labels.csv')
test = pd.read_parquet(f'../input/test.parquet')
cid = pd.Categorical(df.pop('customer_ID'), ordered=True)

df = df.groupby(cid).mean()
print(df.shape)
df.to_parquet(f'../test_input/train.parquet')

print(labels.shape)
labels.to_csv(f'../test_input/train_label.csv')

test = test.groupby(cid).mean()
test = test.head(10000)
print(test.shape)
test.to_parquet(f'../test_input/test.parquet')
