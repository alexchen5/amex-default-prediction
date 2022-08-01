import pandas as pd
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
import matplotlib as plt
from amex_metric import amex_metric

# reading files for train
x_train = pd.read_parquet(f'../test_input/train_lgbm.parquet')
y_train = pd.read_csv(f'../input/train_labels.csv').target.values

def lgb_amex_metric(y_true, y_pred):
    """The competition metric with lightgbm's calling convention"""
    return ('amex',
            amex_metric(y_true, y_pred),
            True)

features = [f for f in x_train.columns if f != 'customer_ID' and f != 'target']

# using the amex metric as the scorer for the grid search
scoring = make_scorer(amex_metric, greater_is_better=True)

# individual tests for different hyperparameters
param_test1 = {
 'max_depth':range(3,10,2),
 'min_child_weight':range(1,6,2)
}
param_test2 = {
 'max_depth':[2,3,4],
 'min_child_weight':[1,2,3]
}
param_test3 = {
 'gamma':[i/10.0 for i in range(0,5)]
}
param_test4 = {
 'subsample':[i/10.0 for i in range(6,10)],
 'colsample_bytree':[i/10.0 for i in range(6,10)]
}
param_test5 = {
 'subsample':[i/100.0 for i in range(75,90,5)],
 'colsample_bytree':[i/100.0 for i in range(50,80,5)]
}

# grid search to optimise hyperparameter provided in param_grid
gsearch1 = GridSearchCV(estimator = xgb.XGBClassifier( learning_rate =0.1, n_estimators=140, max_depth=3,
 min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.75,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=27), 
 param_grid = param_test5, scoring=scoring,n_jobs=4, cv=5)

gsearch1.fit(x_train[features], y_train,eval_metric=lgb_amex_metric)
print(gsearch1.best_params_) 
print(gsearch1.best_score_)