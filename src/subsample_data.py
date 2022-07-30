'''
The goal of this file is to subsample a smaller amount of CIDs from the training 
database.

The results can then be used for data preprocessing.
'''

import random 
import pandas as pd

# x_in: feature set
# y_in: label set related to features, or None
# n_samples: number of CIDs to select
def subsample(x_in: str, x_out: str, y_in=None, y_out=None, n_samples=100):
    df = pd.read_parquet(x_in)
    
    cid = df['customer_ID'].unique()
    cid_sample = cid[random.sample(range(cid.size), n_samples)]
    
    df = df[df['customer_ID'].isin(cid_sample)]
    df.to_parquet(x_out, index=False)
    print(df.shape)
    
    if y_in:
        labels = pd.read_csv(y_in)
        labels = labels[labels['customer_ID'].isin(cid_sample)]
        labels.to_csv(y_out, index=False)
        print(labels.shape)

if __name__ == "__main__":
    dir_in = '../input/amex-data-integer-dtypes-parquet-format'
    dir_out = '../input/subsampled'

    y_train = '../input/amex-default-prediction/train_labels.csv'
    
    subsample(
        x_in=f'{dir_in}/train.parquet', x_out=f'{dir_out}/train.parquet',
        y_in=y_train, y_out=f'{dir_out}/train_labels.csv',
        n_samples=1000
    )
    
    # No related labels to subsample
    subsample(
        x_in=f'{dir_in}/test.parquet', x_out=f'{dir_out}/test.parquet',
        n_samples=2000
    )