import pandas as pd
import lightgbm as lgb

# test = pd.read_parquet(f'../test_input/shrunk_test.parquet')
x_test = pd.read_parquet(f'../input/processed/test.parquet')

lg = lgb.Booster(model_file="../models/lgb_model.json")
# print(lg.predict(test))
pred = lg.predict(x_test)

sub = pd.DataFrame({'customer_ID': x_test.index,
                        'prediction': pred})
sub.to_csv('../output/submission_lgbm.csv', index=False)