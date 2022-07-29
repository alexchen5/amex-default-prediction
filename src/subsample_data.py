'''
The goal of this file is to subsample a smaller amount of CIDs from the training 
database.

The results can then be used for data preprocessing.
'''

import random 
import pandas as pd

n_samples = 100

df = pd.read_parquet(f'../input/amex-data-integer-dtypes-parquet-format/train.parquet')
labels = pd.read_csv(f'../input/amex-default-prediction/train_labels.csv')

cid = df['customer_ID'].unique()
cid_sample = cid[random.sample(range(cid.size), n_samples)]

df = df[df['customer_ID'].isin(cid_sample)]
labels = labels[labels['customer_ID'].isin(cid_sample)]

df.to_parquet(f'../input/subsampled/train.parquet', index=False)
labels.to_csv(f'../input/subsampled/train_labels.csv', index=False)

print(df.shape)
print(labels.shape)
