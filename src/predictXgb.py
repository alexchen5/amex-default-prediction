import pandas as pd
import xgboost as xgb

test = pd.read_parquet(f'../test_input/test_lgbm.parquet')
xg = xgb.XGBClassifier()
xg.load_model(f"../models/xgb_model.json")