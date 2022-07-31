import eli5
from eli5.sklearn import PermutationImportance
import lightgbm as lgb
from matplotlib import axes
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.feature_selection import RFE
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectFromModel
from IPython.display import display

x_train = pd.read_parquet(f'../input/subsample-processed/train.parquet')
y_train = pd.read_csv(f'../input/subsampled/train_labels.csv').target.values

lg = lgb.Booster(model_file="../models/lgb_model.json")

# rfe = RFE(lg, n_features_to_select=1, step=1)
# rfe.fit(x_train, y_train)
# ranking = rfe.ranking_.reshape(x_train.shape)

# # Plot pixel ranking
# plt.matshow(ranking, cmap=plt.cm.Blues)
# plt.colorbar()
# plt.title("Ranking of pixels with RFE")
# plt.show()

print(lg.predict())

perm = PermutationImportance(lg,scoring=None, n_iter=1, random_state=42, cv=None, refit=False)
tmp = eli5.show_weights(perm)
display(eli5.show_weights(perm, top = len(list(x_train.columns)), feature_names = list(x_train.columns)))
