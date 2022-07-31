import pandas as pd
import xgboost as xgb

test = pd.read_parquet(f'../test_input/shrunk_test.parquet')
xg = xgb.XGBClassifier()
xg.load_model(f"../models/xgb_model.json")
print(xg.predict_proba(test)[:,1])