import pandas as pd
import catboost as cb

test = pd.read_parquet(f'../test_input/shrunk_test.parquet')
cb = cb.CatBoostClassifier()
cb.load_model("../models/cb_model.json")
print(cb.predict_proba(test)[:,1])