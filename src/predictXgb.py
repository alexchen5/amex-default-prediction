import pandas as pd
import xgboost as xgb
from amex_metric import amex_metric
import numpy as np

# predicting with previously saved xgb model on test data
test = pd.read_parquet(f'../input/processed/train.parquet')
test_labels = pd.read_csv(f'../input/amex-default-prediction/train_labels.csv').target.values

xg = xgb.XGBClassifier()
xg.load_model(f"../models/xgb_model.json")
pred = xg.predict_proba(test)[:,1]

print(np.mean(amex_metric(test_labels, pred)))

sub = pd.DataFrame({'customer_ID': test.index,
                        'prediction': pred})
sub.to_csv('../output/submission_xgb.csv', index=False)