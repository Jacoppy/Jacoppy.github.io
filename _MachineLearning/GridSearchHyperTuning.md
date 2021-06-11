---
title: "Hyperparameter Tuning GDBT"
search: true
categories: 
  - MachineLearning
---

Hyperparameters tuning represents a necessity while working with AI models in order to optimize the performances. There are different approaches to this: in this little article I am going to cover the bruteforce one, GridSearchCV.

# What is GridSearchCV?

- GridSearch : process of performing hyperparameter tuning in order to determine the optimal values for a given model. 
- CV : Cross-Valication
- How does it work? GridSearchCV tries all the combinations of the values passed in the dictionary and evaluates the model for each combination using the Cross-Validation method. Hence after using this function we get accuracy/loss for every combination of hyperparameters and we can choose the one with the best performance.

# Parameters to tune

## XGBoost

- subsample  :  Each tree will only get a % of the training examples and can be values between 0 and 1. Lowering this value stops subsets of training examples dominating the model and allows greater generalisation.
- colsample_bytree  : Similar to subsample but for columns rather than rows. Again you can set values between 0 and 1 where lower values can make the model generalise better by stopping any one field having too much prominence, a prominence that might not exist in the test data.
- colsample_bylevel  : Denotes the fraction of columns to be randomly samples for each tree.
- n_estimators : is the number of iterations the model will perform or in other words the number of trees that will be created
- learning_rate : in layman’s terms it is how much the weights are adjusted each time a tree is built. Set the learning rate too high and the algorithm might miss the optimum weights but set it too low and it might converge to suboptimal values.



## LighGBM

- subsample [0-1] :  Each tree will only get a % of the training examples and can be values between 0 and 1. Lowering this value stops subsets of training examples dominating the model and allows greater generalisation.
- colsample_bytree [0-1] : Similar to subsample but for columns rather than rows. Again you can set values between 0 and 1 where lower values can make the model generalise better by stopping any one field having too much prominence, a prominence that might not exist in the test data.
- colsample_bylevel [0-1] : Denotes the fraction of columns to be randomly samples for each tree.
- n_estimators : is the number of iterations the model will perform or in other words the number of trees that will be created
- learning_rate : in layman’s terms it is how much the weights are adjusted each time a tree is built. Set the learning rate too high and the algorithm might miss the optimum weights but set it too low and it might converge to suboptimal values.
- num_leaves :you set the maximum number of leaves each weak learner has: large num_leaves increases accuracy on the training set and also the chance of getting hurt by overfitting
- feature_fraction : deals with column sampling and can be used to speed up training or deal with overfitting
- reg_alpha, reg_alpha : regularization 

# Code 

```
import xgboost as xgb
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from sklearn.preprocessing import normalize
import ujson as json
from settings import config
import pickle
from sklearn.feature_extraction import DictVectorizer
import numpy as np
import lightgbm as lgb

def pipeline_GridSearch(X_train_data , X_test_data , y_train_data ,
                       model , param_grid , cv=10 , scoring_fit='roc_auc' ,
                       do_probabilities=False) :
    gs = GridSearchCV (
        estimator=model ,
        param_grid=param_grid ,
        cv=cv ,
        n_jobs=-1 ,
        scoring=scoring_fit ,
        verbose=2
    )
    fitted_model = gs.fit ( X_train_data , y_train_data )

    if do_probabilities :
        pred = fitted_model.predict_proba ( X_test_data )
    else :
        pred = fitted_model.predict ( X_test_data )

    return fitted_model , pred





X,y,X_train, X_test, y_train, y_test, X_validation, y_validation, vec =load_features_drebin(config['Drebin_X_file'],config['Drebin_Y_file'])
xgb_Classifier = xgb.XGBClassifier(
    objective= 'binary:logistic',
    nthread=4,
    seed=42
)
#GridSearch
gbm_param_grid = {
    'colsample_bytree': np.linspace(0.1, 0.5, 5),
    'subsamples':np.linspace(0.1, 1.0, 10),
    'colsample_bylevel':np.linspace(0.1, 0.5, 5),
    'n_estimators': list(range(60, 340, 40)),
    'max_depth': list(range(2,16)),
    'learning_rate':np.logspace(-3, -0.8, 5)
}
model_xgb,preds = pipeline_GridSearch(X_train, X_test, y_train, xgb_Classifier,gbm_param_grid, cv=5, scoring_fit='roc_auc')

fixed_params = {'objective': 'binary',
             'metric': 'auc',
             'is_unbalance':True,
             'bagging_freq':5,
             'boosting':'dart',
             'num_boost_round':300,
             'early_stopping_rounds':30}
lgb_classifier = lgb.LGBMClassifier()
lgb_param_grid = {
    'n_estimators': list(range(60, 340, 40)),
    'colsample_bytree': np.linspace(0.1, 0.5, 5),
    'max_depth': list(range(2,16)),
    'num_leaves':list(range(50, 200, 50)),
    'reg_alpha': np.logspace(-3, -2, 3),
    'reg_lambda': np.logspace(-2, 1, 4),
    'subsample': np.linspace(0.1, 1.0, 10),
    'feature_fraction': np.linspace(0.1, 1.0, 10),
    'learning_rate':np.logspace(-3, -0.8, 5)
}
model_lgbm,preds = pipeline_GridSearch(X_train, X_test, y_train, lgb_classifier,lgb_param_grid, cv=5, scoring_fit='roc_auc')
print("Grid Search Best parameters found LGBM: ", model_lgbm.best_params_)

print("Grid Search Best parameters found XGB: ", model_xgb.best_params_)

```

There are also other ways to tune hyperparameters of course :)
E.g : [Bayesian optimization](https://towardsdatascience.com/beyond-grid-search-hypercharge-hyperparameter-tuning-for-xgboost-7c78f7a2929d)

Happy tuning!