import datetime
from colorama import Fore, Style
import pandas as pd
import numpy as np
from scipy.stats import skew
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LogisticRegression
from math import sqrt
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier, log_evaluation
import gc
from amex_metric import amex_metric
import matplotlib.pyplot as plt
from sklearn.ensemble import StackingClassifier
NFOLDS = 5
SEED = 0
NROWS = None

x_train = pd.read_parquet(f'../test_input/train_lgbm.parquet')
y_train = pd.read_csv(f'../test_input/train_label.csv').target.values
# x_test = pd.read_parquet(f'../test_input/test.parquet')

features = [f for f in x_train.columns if f != 'customer_ID' and f != 'target']
# x_train = x_train.fillna(0)
# x_test= x_test.fillna(0)
ntrain = x_train.shape[0]
# ntest = x_test.shape[0]
# print(ntest)
kf = KFold(n_splits = NFOLDS, shuffle=True, random_state=SEED)

def lgb_amex_metric(y_true, y_pred):
    """The competition metric with lightgbm's calling convention"""
    return ('amex',
            amex_metric(y_true, y_pred),
            True)

class SklearnWrapper(object):
    def __init__(self, clf, seed=0, params=None):
        params['random_state'] = seed
        self.clf = clf(**params)

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train, eval_metric=[lgb_amex_metric])

    def predict(self, x):
        return self.clf.predict_proba(x)[:,1]

class CatboostWrapper(object):
    def __init__(self, clf, seed=0, params=None):
        params['random_seed'] = seed
        self.clf = clf(**params)

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict_proba(x)[:,1]
    
    def save_model(self):
        return self.clf.save_model(f'../models/cb_model.json')
        
class LightGBMWrapper(object):
    def __init__(self, clf, params=None):
        # params['feature_fraction_seed'] = seed
        # params['bagging_seed'] = seed
        self.clf = clf(**params)

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict_proba(x)[:,1]

    def save_model(self):
        return self.clf.booster_.save_model(f'../models/lgb_model.json')


class XgbWrapper(object):
    def __init__(self, seed=0, params=None):
        self.param = params
        self.param['seed'] = seed
        self.nrounds = params.pop('nrounds', 250)

    def train(self, x_train, y_train):
        dtrain = xgb.DMatrix(x_train, label=y_train)
        self.gbdt = xgb.train(self.param, dtrain, self.nrounds)

    def predict(self, x):
        return self.gbdt.predict(xgb.DMatrix(x))

    def save_model(self):
        return self.gbdt.save_model(f'../models/xgb_model.json')

def get_oof(clf):
    score_list = []
    y_pred_list = []
    kf = StratifiedKFold(n_splits=5)
    for fold, (idx_tr, idx_va) in enumerate(kf.split(x_train, y_train)):
        X_tr, X_va, y_tr, y_va= None, None, None, None
        start_time = datetime.datetime.now()
        X_tr = x_train.iloc[idx_tr][features]
        X_va = x_train.iloc[idx_va][features]
        y_tr = y_train[idx_tr]
        y_va = y_train[idx_va]
    
        clf.train(X_tr, y_tr)
        X_tr, y_tr = None, None
        y_va_pred = clf.predict(X_va)
        score = amex_metric(y_va, y_va_pred)
        # n_trees = clf.best_iteration_
        # if n_trees is None: n_trees = clf.n_estimators
        # print(f"{Fore.GREEN}{Style.BRIGHT}Fold {fold} | {str(datetime.datetime.now() - start_time)[-12:-7]} |"
        #     f" {n_trees:5} trees |"
        #     f"                Score = {score:.5f}{Style.RESET_ALL}")
        score_list.append(score)
        
        # if INFERENCE:
        y_pred_list.append(clf.predict(x_train[features]))
            
        # if ONLY_FIRST_FOLD: break # we only want the first fold
    
    print(f"{Fore.GREEN}{Style.BRIGHT}OOF Score:                       {np.mean(score_list):.5f}{Style.RESET_ALL}")
    return np.mean(y_pred_list, axis=0), np.mean(score_list)
et_params = {
    'n_jobs': 16,
    'n_estimators': 200,
    'max_features': 0.5,
    'max_depth': 12,
    'min_samples_leaf': 2,
}

rf_params = {
    'n_jobs': 16,
    'n_estimators': 200,
    'max_features': 0.2,
    'max_depth': 12,
    'min_samples_leaf': 2,
}

xgb_params = {
    'seed': 0,
    'colsample_bytree': 0.6,
    'subsample': 0.8,
    'learning_rate': 0.05,
    'objective': 'binary:logistic',
    'eval_metric':'logloss',
    'max_depth': 4,
    'num_parallel_tree': 1,
    'min_child_weight': 1,
    'nrounds': 200
}

catboost_params = {
    'iterations': 250,
    'learning_rate': 0.5,
    'depth': 3,
    'l2_leaf_reg': 40,
    'bootstrap_type': 'Bernoulli',
    'subsample': 0.7,
    'scale_pos_weight': 5,
    'eval_metric': 'AUC',
    'od_type': 'Iter',
    'allow_writing_files': False
}

lightgbm_params = {
    'n_estimators': 1200,
    'learning_rate': 0.03, 'reg_lambda':50,
    'min_child_samples':2400,
    'num_leaves':95,
    'colsample_bytree':0.19,
    'max_bins':511, 'random_state':1  
}


xg = XgbWrapper(seed=SEED, params=xgb_params)
et = SklearnWrapper(clf=ExtraTreesClassifier, seed=SEED, params=et_params)
rf = SklearnWrapper(clf=RandomForestClassifier, seed=SEED, params=rf_params)
cb = CatboostWrapper(clf= CatBoostClassifier, seed = SEED, params=catboost_params)
lg = LightGBMWrapper(clf = LGBMClassifier, params = lightgbm_params)


# et_oof_train, et_oof_test = get_oof(et)
# rf_oof_train, rf_oof_test = get_oof(rf)
lg_oof_train, lg_score = get_oof(lg)
cb_oof_train, cb_score = get_oof(cb)
x_train = x_train.fillna(0)
xg_oof_train, xg_score = get_oof(xg)
# cb_oof_train, cb_score = get_oof(cb)

# print("XG-CV: {}".format(xg_score))
# xg.save_model()
# print("LG-CV: {}".format(lg_score))
# lg.save_model()
# print("CB-CV: {}".format(cb_score))
# cb.save_model()
# print(cb_oof_train.shape)
# x_train = np.hstack((xg_oof_train,lg_oof_train, cb_oof_train)).reshape(-1, 3)
# x_train = np.concatenate((xg_oof_train, lg_oof_train, cb_oof_train), axis=1)
# gc.collect()
# print(x_train.shape)
# print(y_train.shape)

xg_pred = xg.predict(x_train)
print(np.mean(amex_metric(y_train, xg_pred)))
lg_pred = lg.predict(x_train)
print(np.mean(amex_metric(y_train, lg_pred)))
cb_pred = cb.predict(x_train)
print(np.mean(amex_metric(y_train, cb_pred)))
final_pred = 0.325 * xg_pred + 0.325 * cb_pred + 0.35 * lg_pred
print(np.mean(amex_metric(y_train, final_pred)))
# final_pred = 0.5 * cb_pred + 5 * lg_pred

# sub = pd.DataFrame({'customer_ID': x_test.index,
#                         'prediction': final_pred})
# sub.to_csv('../output/submission_stacked.csv', index=False)

# sub = pd.DataFrame({'customer_ID': x_test.index,
#                         'prediction': xg_pred})
# sub.to_csv('../output/submission_xg.csv', index=False)

# sub = pd.DataFrame({'customer_ID': x_test.index,
#                         'prediction': lg_pred})
# sub.to_csv('../output/submission_lg.csv', index=False)

# sub = pd.DataFrame({'customer_ID': x_test.index,
#                         'prediction': cb_pred})
# sub.to_csv('../output/submission_cb.csv', index=False)


# logistic_regression = LogisticRegression(max_iter=10000000)
# logistic_regression.fit(x_train,y_train,)
# y_pred_train = logistic_regression.predict_proba(x_train)
# y_pred_test = logistic_regression.predict_proba(x_test)
# print("Finished predicting")
# # print(x_train)
# # print(y_train.shape)
# print(amex_metric(y_train, y_pred_train[:,1]))
# print(y_pred_test[:,1].shape)
# sub = pd.DataFrame({'customer_ID': x_test.index,
#                         'prediction': y_pred_test[:,1]})
# sub.to_csv('../output/submission_lgbm.csv', index=False)
# lr = LogisticRegression()
# stack_model = StackingClassifier( estimators = clf,final_estimator = lr)
# y_va_pred = stack_model.predict_proba()
# score = amex_metric(y_va, y_va_pred)
