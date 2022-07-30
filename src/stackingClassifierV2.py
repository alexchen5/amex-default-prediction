from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB 
from sklearn.ensemble import RandomForestClassifier
from mlxtend.classifier import StackingCVClassifier
import numpy as np
import warnings
import pandas as pd
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import xgboost as xgb
from amex_metric import amex_metric
from sklearn.metrics import make_scorer

def lgb_amex_metric(y_true, y_pred):
    """The competition metric with lightgbm's calling convention"""
    return ('amex',
            amex_metric(y_true, y_pred),
            True)

warnings.simplefilter('ignore')

RANDOM_SEED = 42

x_train = pd.read_parquet(f'../test_input/shrunk_train.parquet')
y_train = pd.read_csv(f'../test_input/shrunk_train_label.csv').target.values
x_test = pd.read_parquet(f'../test_input/shrunk_test.parquet')

clf1 = LGBMClassifier(n_estimators= 1200,
    learning_rate= 0.03, reg_lambda=50,
    min_child_samples=2400,
    num_leaves=95,
    colsample_bytree=0.19,
    max_bins=511, random_state=1 )
clf2 = CatBoostClassifier( iterations= 200,
    learning_rate= 0.5,
    depth= 3,
    l2_leaf_reg= 40,
    bootstrap_type= 'Bernoulli',
    subsample= 0.7,
    scale_pos_weight= 5,
    eval_metric= 'AUC',
    od_type= 'Iter',
    allow_writing_files= False)
lr = LogisticRegression()

scoring = make_scorer(amex_metric, greater_is_better=True)

# Starting from v0.16.0, StackingCVRegressor supports
# `random_state` to get deterministic result.
sclf = StackingCVClassifier(classifiers=[clf1, clf2],
                            meta_classifier=lr,
                            random_state=RANDOM_SEED)

print('3-fold cross validation:\n')
accuracy = []
for clf, label in zip([clf1, clf2, sclf], 
                      ['LGBM', 
                       'CatBoost', 
                       'StackingClassifier']):

      scores = model_selection.cross_val_score(clf, x_train, y_train, 
                                                cv=3, scoring=[lgb_amex_metric])
      accuracy.append(scores)
      
print(np.mean(accuracy, axis=0))
sclf.fit(x_train, y_train)
y_pred = sclf.predict_proba(x_train)
print(y_pred[:, 0].shape)