from math import gamma
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from amex_metric import amex_metric

# reading in train and test data
x_train = pd.read_parquet(f'../input/processed/train.parquet')
y_train = pd.read_csv(f'../input/amex-default-prediction/train_labels.csv').target.values
x_test = pd.read_parquet(f'../input/processed/test.parquet')

features = [f for f in x_train.columns if f != 'customer_ID' and f != 'target']

# metric used by competition for scoring
def lgb_amex_metric(y_true, y_pred):
    """The competition metric with lightgbm's calling convention"""
    return ('amex',
            amex_metric(y_true, y_pred),
            True)

# wrapper classes for Catboost, LightGBM and XGBoost to allow for all models to be trained with cross validation with one function
class CatboostWrapper(object):
    def __init__(self, clf, seed=0, params=None):
        params['random_seed'] = seed
        self.clf = clf(**params)

    def fit(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict_proba(x)[:,1]
    
    def save_model(self):
        return self.clf.save_model(f'../models/cb_model.json')
        
class LightGBMWrapper(object):
    def __init__(self, clf, params=None):
        self.clf = clf(**params)

    def fit(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict_proba(x)[:,1]

    def save_model(self):
        return self.clf.booster_.save_model(f'../models/lgb_model.json')

class XgbWrapper(object):
    def __init__(self, params=None):
        self.param = params
        self.nrounds = params.pop('nrounds', 250)

    def fit(self, x_train, y_train):
        dtrain = xgb.DMatrix(x_train, label=y_train)
        self.gbdt = xgb.train(self.param, dtrain, self.nrounds)

    def predict(self, x):
        return self.gbdt.predict(xgb.DMatrix(x))

    def save_model(self):
        return self.gbdt.save_model(f'../models/xgb_model.json')

# function that calculated the out of fold score for the provided model which is taken as the CV score
def get_oof(clf):
    scores = []
    predictions = []
    kf = StratifiedKFold(n_splits=5)
    for fold, (idx_train, idx_valid) in enumerate(kf.split(x_train, y_train)):
        X_train, X_valid, y_train, y_valid= None, None, None, None
        X_train = x_train.iloc[idx_train][features]
        X_valid = x_train.iloc[idx_valid][features]
        y_train = y_train[idx_train]
        y_valid = y_train[idx_valid]
    
        clf.fit(X_train, y_train)
        X_train, y_train = None, None
        y_va_pred = clf.predict(X_valid)
        score = amex_metric(y_valid, y_va_pred)
        scores.append(score)
        predictions.append(clf.predict(x_train[features]))
    
    print(f"OOF Score:{np.mean(scores)}")
    return np.mean(scores)

# hyperparameters for XGBoost, Catboost and LightGBM, respectively
xgb_params = {
    'seed': 0,
    'colsample_bytree': 0.75,
    'subsample': 0.8,
    'learning_rate': 0.05,
    'objective': 'binary:logistic',
    'eval_metric':'logloss',
    'max_depth': 3,
    'num_parallel_tree': 1,
    'min_child_weight': 1,
    'nrounds': 200,
    'gamma': 0,
}

catboost_params = {
    'iterations': 1000,
    'learning_rate': 0.2,
    'depth': 5,
    'l2_leaf_reg': 3,
    'bootstrap_type': 'Bernoulli',
    'subsample': 0.7,
    'scale_pos_weight': 5,
    'eval_metric': 'AUC',
    'od_type': 'Iter',
    'allow_writing_files': False,
    'random_seed': 0,
    'border_count': 20
}

lightgbm_params = {
    'n_estimators': 1200,
    'learning_rate': 0.03, 
    'reg_lambda':50,
    'min_child_samples':439,
    'num_leaves':46,
    'colsample_bytree':0.47,
    'max_bins':511, 
    'random_state':1,
    'subsample': 0.2,
    'reg_alpha': 1,
    'min_child_weight': 1  
}


xg = XgbWrapper(seed=0, params=xgb_params)
cb = CatboostWrapper(clf= CatBoostClassifier, seed = 0, params=catboost_params)
lg = LightGBMWrapper(clf = LGBMClassifier, params = lightgbm_params)

lg_score = get_oof(lg)
cb_score = get_oof(cb)

# Fill na for models which need it
x_train = x_train.fillna(0)
x_test= x_test.fillna(0)

xg_score = get_oof(xg)

print("XG-CV: {}".format(xg_score))
xg.save_model()
print("LG-CV: {}".format(lg_score))
lg.save_model()
print("CB-CV: {}".format(cb_score))
cb.save_model()

xg_pred = xg.predict(x_train)
print(np.mean(amex_metric(y_train, xg_pred)))
lg_pred = lg.predict(x_train)
print(np.mean(amex_metric(y_train, lg_pred)))
cb_pred = cb.predict(x_train)
print(np.mean(amex_metric(y_train, cb_pred)))
final_pred = 0.325 * xg_pred + 0.325 * cb_pred + 0.35 * lg_pred
print(np.mean(amex_metric(y_train, final_pred)))

# predictions on the test data using trained models
xg_pred = xg.predict(x_test)
print(xg_pred)
lg_pred = lg.predict(x_test)
print(lg_pred)
cb_pred = cb.predict(x_test)
print(cb_pred)
final_pred = 0.325 * xg_pred + 0.325 * cb_pred + 0.35 * lg_pred

# saving the predictions into csv files
sub = pd.DataFrame({'customer_ID': x_test.index,
                        'prediction': final_pred})
sub.to_csv('../output/submission_stacked.csv', index=False)

sub = pd.DataFrame({'customer_ID': x_test.index,
                        'prediction': xg_pred})
sub.to_csv('../output/submission_xg.csv', index=False)

sub = pd.DataFrame({'customer_ID': x_test.index,
                        'prediction': lg_pred})
sub.to_csv('../output/submission_lg.csv', index=False)

sub = pd.DataFrame({'customer_ID': x_test.index,
                        'prediction': cb_pred})
sub.to_csv('../output/submission_cb.csv', index=False)

