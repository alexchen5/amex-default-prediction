# compare ensemble to each baseline classifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import StackingClassifier
from matplotlib import pyplot
import pandas as pd
from amex_metric import amex_metric

scoring = make_scorer(amex_metric, greater_is_better=True)

# get a stacking ensemble of models
def get_stacking():
	# define the base models
	level0 = list()
	level0.append(('lr', LogisticRegression()))
	level0.append(('lg', LGBMClassifier()))
	level0.append(('cb', CatBoostClassifier()))
	# define meta learner model
	level1 = LogisticRegression()
	# define the stacking ensemble
	model = StackingClassifier(estimators=level0, final_estimator=level1, cv=5)
	return model
 
# get a list of models to evaluate
def get_models():
	models = dict()
	models['lr'] = LogisticRegression()
	models['lg'] = LGBMClassifier()
	models['cb'] = CatBoostClassifier()
	models['stacking'] = get_stacking()
	return models
 
# evaluate a give model using cross-validation
def evaluate_model(model, X, y):
	cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
	scores = cross_val_score(model, X, y, scoring=scoring, cv=cv, n_jobs=-1, error_score='raise')
	return scores
 
# define dataset
X = pd.read_parquet(f'../test_input/shrunk_train.parquet')
y = pd.read_csv(f'../test_input/shrunk_train_label.csv').target.values
# train = pd.read_parquet(f'../test_input/shrunk_train.parquet')
# target = pd.read_csv(f'../test_input/shrunk_train_label.csv').target.values
X = X.fillna(-127)
# X_test = X_test.fillna(-127)
# get the models to evaluate
models = get_models()
# evaluate the models and store results
results, names = list(), list()
for name, model in models.items():
	scores = evaluate_model(model, X, y)
	results.append(scores)
	names.append(name)
	print('>%s %.3f (%.3f)' % (name, mean(scores), std(scores)))
# plot model performance for comparison
pyplot.boxplot(results, labels=names, showmeans=True)
pyplot.show()