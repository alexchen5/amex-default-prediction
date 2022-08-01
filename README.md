# amex-default-prediction
Our team's solution to a Machine Learning Comp hosted by American Express. 

## Preprocessing and sampling
* subsample_data.py: Randomly selects CIDS and all of their monthly statements from the provided data
* shrinkData.py: Averages the monthly statements of each CID and returns a specified amount of that data
* preprocess.py: Performs feature engineering as described in report

## Metric
* amex_metric.py: Contains the calculations for the metric used for scoring in the competition

## Model Training
* modelTrain.py: Trains the three models specified within report (XGBoost, Catboost, LightGBM), providing CV score and saves these models

## Prediction
* predictLgb.py: Loads provided lgb model and makes prediction on the provided data
* predictXgb.py: Loads provided Xgb model and makes prediction on the provided data
* predictCb.py: Loads provided Cb model and makes prediction on the provided data

## Hyperparameter Tuning
* lgbmTuning.py: Tunes the hyperparameters of a LGBM model
* xgbTuning.py: Tunes the hyperparameters of a LGBM model
* cbTuning.py Tunes the hyperparameters of a LGBM model

## Processed Dataset
* [Processed Data](https://www.kaggle.com/datasets/alxchen5/amex-processed?fbclid=IwAR3v9FMl69YjK6xl59iXdYyV02Yva-u6hV9ZYrVpCz3vbL-FlP8bptiLsRc)

## Future Works
* permutation_importance.py: Permutation importance to improve on dataset processing