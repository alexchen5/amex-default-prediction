import pandas as pd
import lightgbm as lgb
from amex_metric import amex_metric
import numpy as np

# test = pd.read_parquet(f'../test_input/shrunk_test.parquet')

test = pd.read_parquet(f'../input/processed/train.parquet')
test_labels = pd.read_csv(f'../input/amex-default-prediction/train_labels.csv').target.values
# test = pd.read_parquet(f'../input/processed/test.parquet')

lg = lgb.Booster(model_file="../models/lgb_model.json")
pred = lg.predict(test)

print(np.mean(amex_metric(test_labels, pred)))

# sub = pd.DataFrame({'customer_ID': x_test.index,
#                         'prediction': pred})
# sub.to_csv('../output/submission_lgbm.csv', index=False)