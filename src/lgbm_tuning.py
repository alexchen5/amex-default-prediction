import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from importData import import_data
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import auc, mean_absolute_error, make_scorer
from amex_metric import amex_metric
import time
from datetime import timedelta
start_time = time.monotonic()

X = pd.read_parquet(f'../test_input/train_lgbm.parquet')
y = pd.read_csv(f'../input/train_labels.csv').target.values

def lgb_amex_metric(y_true, y_pred):
    """The competition metric with lightgbm's calling convention"""
    return ('amex',
            amex_metric(y_true, y_pred),
            True)

def learning_rate_010_decay_power_099(current_iter):
    base_learning_rate = 0.1
    lr = base_learning_rate  * np.power(.99, current_iter)
    return lr if lr > 1e-3 else 1e-3

def learning_rate_010_decay_power_0995(current_iter):
    base_learning_rate = 0.1
    lr = base_learning_rate  * np.power(.995, current_iter)
    return lr if lr > 1e-3 else 1e-3

def learning_rate_005_decay_power_099(current_iter):
    base_learning_rate = 0.05
    lr = base_learning_rate  * np.power(.99, current_iter)
    return lr if lr > 1e-3 else 1e-3

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=314)

fit_params={"early_stopping_rounds":30, 
        "eval_metric" : [lgb_amex_metric], 
        "eval_set" : [(X_test,y_test)],
        'eval_names': ['valid'],
        #'callbacks': [lgb.reset_parameter(learning_rate=learning_rate_010_decay_power_099)],
        'categorical_feature': 'auto'}

param_test ={'num_leaves': sp_randint(6, 50), 
             'min_child_samples': sp_randint(100, 500), 
             'min_child_weight': [1e-5, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4],
             'subsample': sp_uniform(loc=0.2, scale=0.8), 
             'colsample_bytree': sp_uniform(loc=0.4, scale=0.6),
             'reg_alpha': [0, 1e-1, 1, 2, 5, 7, 10, 50, 100],
             'reg_lambda': [0, 1e-1, 1, 5, 10, 20, 50, 100]}

n_HP_points_to_test = 100

scoring = make_scorer(amex_metric, greater_is_better=True)
# metric=[lgb_amex_metric]
clf = lgb.LGBMClassifier(max_depth=-1, random_state=314, n_jobs=4, n_estimators=5000)
gs = RandomizedSearchCV(
    estimator=clf, param_distributions=param_test, 
    n_iter=n_HP_points_to_test,
    scoring=scoring,
    cv=3,
    refit=True,
    random_state=314)

gs.fit(X_train, y_train, **fit_params)
print('Best score reached: {} with params: {} '.format(gs.best_score_, gs.best_params_))

end_time = time.monotonic()
print(timedelta(seconds=end_time - start_time))