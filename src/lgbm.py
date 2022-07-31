import lightgbm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import ListedColormap
from cycler import cycler
from IPython.display import display
import datetime
import scipy.stats
import warnings
from colorama import Fore, Back, Style
import gc
from amex_metric import amex_metric

from sklearn.model_selection import StratifiedKFold
from sklearn.calibration import CalibrationDisplay
from lightgbm import LGBMClassifier, log_evaluation

plt.rcParams['axes.facecolor'] = '#0057b8' # blue
plt.rcParams['axes.prop_cycle'] = cycler(color=['#ffd700'] +
                                         plt.rcParams['axes.prop_cycle'].by_key()['color'][1:])
plt.rcParams['text.color'] = 'w'

INFERENCE = True # set to False if you only want to cross-validate

def lgb_amex_metric(y_true, y_pred):
    """The competition metric with lightgbm's calling convention"""
    return ('amex',
            amex_metric(y_true, y_pred),
            True)

train = pd.read_parquet(f'../input/processed/train.parquet')  
test = pd.read_parquet(f'../input/processed/test.parquet')  
target = pd.read_csv('../input/amex-default-prediction/train_labels.csv').target.values

# train = pd.read_parquet(f'../input/subsampled/train.parquet')
# test = pd.read_parquet(f'../input/subsampled/test.parquet')
# target = pd.read_csv('../input/subsampled/train_labels.csv').target.values

print(f"train shape: {train.shape}")
print(f"test shape: {test.shape}")
print(f"target shape: {target.shape}")

def my_booster(random_state=1, n_estimators=1200):
    return LGBMClassifier(n_estimators=n_estimators,
                          learning_rate=0.03, reg_lambda=50,
                          min_child_samples=2400,
                          num_leaves=46,
                          reg_alpha=1,
                          min_child_weight=1,
                          colsample_bytree=0.46717,
                          subsample=0.2065,
                          max_bins=511, random_state=random_state)
model = my_booster()
model.fit(train, target,
    eval_metric=[lgb_amex_metric],
    callbacks=[log_evaluation(100)])

train_predict = model.predict_proba(train, raw_score=True)
train_score = amex_metric(target, train_predict)
print(train_score)

test_predict = model.predict_proba(test, raw_score=True)
sub = pd.DataFrame({'customer_ID': test.index,
                    'prediction': test_predict})
sub.to_csv('../output/submission_lgbm.csv', index=False)

exit()
# Cross-validation of the classifier

ONLY_FIRST_FOLD = False

features = [f for f in train.columns if f != 'customer_ID' and f != 'target']

def my_booster(random_state=1, n_estimators=1200):
    return LGBMClassifier(n_estimators=n_estimators,
                          learning_rate=0.03, reg_lambda=50,
                          min_child_samples=2400,
                          num_leaves=46,
                          reg_alpha=1,
                          min_child_weight=1,
                          colsample_bytree=0.46717,
                          subsample=0.2065,
                          max_bins=511, random_state=random_state)

fig, ax = plt.subplots(1, figsize = (10,5))

print(f"{len(features)} features")
score_list = []
y_pred_list = []
kf = StratifiedKFold(n_splits=5)
for fold, (idx_tr, idx_va) in enumerate(kf.split(train, target)):
    X_tr, X_va, y_tr, y_va, model = None, None, None, None, None
    start_time = datetime.datetime.now()
    X_tr = train.iloc[idx_tr][features]
    X_va = train.iloc[idx_va][features]
    y_tr = target[idx_tr]
    y_va = target[idx_va]
    
    model = my_booster()
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=UserWarning)
        model.fit(X_tr, y_tr,
                  eval_set = [(X_va, y_va)], 
                  eval_metric=[lgb_amex_metric],
                  callbacks=[log_evaluation(100)])
    X_tr, y_tr = None, None
    y_va_pred = model.predict_proba(X_va, raw_score=True)
    score = amex_metric(y_va, y_va_pred)
    n_trees = model.best_iteration_
    if n_trees is None: n_trees = model.n_estimators
    print(f"{Fore.GREEN}{Style.BRIGHT}Fold {fold} | {str(datetime.datetime.now() - start_time)[-12:-7]} |"
          f" {n_trees:5} trees |"
          f"                Score = {score:.5f}{Style.RESET_ALL}")
    score_list.append(score)
    lightgbm.plot_metric(model, metric = 'amex', ax=ax)

    # if INFERENCE:
    #     # y_pred_list.append(model.predict_proba(test[features], raw_score=True))
        
    # if ONLY_FIRST_FOLD: break # we only want the first fold
    
print(f"{Fore.GREEN}{Style.BRIGHT}OOF Score:                       {np.mean(score_list):.5f}{Style.RESET_ALL}")
def sigmoid(log_odds):
    return 1 / (1 + np.exp(-log_odds))

# if INFERENCE:
#     sub = pd.DataFrame({'customer_ID': test.index,
#                         'prediction': np.mean(y_pred_list, axis=0)})
#     sub.to_csv('../output/submission_lgbm.csv', index=False)
ax.legend(labels=["Fold 1","Fold 2","Fold 3","Fold 4","Fold 5"])
ax.set_facecolor("white")
ax.set_title("Amex metric score of Lightgbm for each fold")
plt.show()