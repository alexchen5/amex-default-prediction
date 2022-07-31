import pandas as pd
import lightgbm as lgb

test = pd.read_parquet(f'../test_input/shrunk_test.parquet')
lg = lgb.Booster(model_file="../models/lgb_model.json")
print(lg.predict(test))