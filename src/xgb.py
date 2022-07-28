# LOAD LIBRARIES
import pandas as pd, numpy as np # CPU libraries
# import cupy, cudf # GPU libraries
import matplotlib.pyplot as plt, gc, os

# print('RAPIDS version',cudf.__version__)

# VERSION NAME FOR SAVED MODEL FILES
VER = 1

# TRAIN RANDOM SEED
SEED = 42

# FILL NAN VALUE
NAN_VALUE = -127 # will fit in int8

# FOLDS PER MODEL
FOLDS = 5

def read_file(path = '', usecols = None):
    # LOAD DATAFRAME
    if usecols is not None: df = pd.read_parquet(path, columns=usecols)
    else: df = pd.read_parquet(path)
    # REDUCE DTYPE FOR DATE
    df.S_2 = pd.to_datetime( df.S_2 )
    # SORT BY CUSTOMER AND DATE (so agg('last') works correctly)
    # df = df.sort_values(['customer_ID','S_2'])
    # df = df.reset_index(drop=True)
    # FILL NAN
    df = df.fillna(NAN_VALUE) 
    print('shape of data:', df.shape)
    
    return df

train = read_file(f'../input/train.parquet')
test = read_file(f'../input/test.parquet')

print(train.head())
