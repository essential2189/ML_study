import gc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

train = pd.read_csv('input/tabular-playground-series-oct-2021/train.csv')
test = pd.read_csv('input/tabular-playground-series-oct-2021/test.csv')
train['std'] = train.std(axis=1)
train['min'] = train.min(axis=1)
train['max'] = train.max(axis=1)

test['std'] = test.std(axis=1)
test['min'] = test.min(axis=1)
test['max'] = test.max(axis=1)
y = train['target']
train.drop(columns=['id', 'target'], inplace=True)
test.drop(columns='id', inplace=True)


def Stacking_Data_Loader(model, model_name, train, y, test, fold):
    '''
    Put your train, test datasets and fold value!
    This function returns train, test datasets for stacking ensemble :)
    '''

    stk = StratifiedKFold(n_splits=fold, random_state=42, shuffle=True)

    # Declaration Pred Datasets
    train_fold_pred = np.zeros((train.shape[0], 1))
    test_pred = np.zeros((test.shape[0], fold))

    for counter, (train_index, valid_index) in enumerate(stk.split(train, y)):
        x_train, y_train = train.iloc[train_index], y[train_index]
        x_valid, y_valid = train.iloc[valid_index], y[valid_index]

        print('------------ Fold', counter + 1, 'Start! ------------')
        if model_name == 'cat':
            model.fit(x_train, y_train, eval_set=[(x_valid, y_valid)])
        elif model_name == 'xgb':
            model.fit(x_train, y_train, eval_set=[(x_valid, y_valid)], eval_metric='auc', verbose=500,
                      early_stopping_rounds=200)
        else:
            model.fit(x_train, y_train, eval_set=[(x_valid, y_valid)], eval_metric='auc', verbose=500,
                      early_stopping_rounds=200)

        print('------------ Fold', counter + 1, 'Done! ------------')

        train_fold_pred[valid_index, :] = model.predict_proba(x_valid)[:, 1].reshape(-1, 1)
        test_pred[:, counter] = model.predict_proba(test)[:, 1]

        del x_train, y_train, x_valid, y_valid
        gc.collect()

    test_pred_mean = np.mean(test_pred, axis=1).reshape(-1, 1)

    del test_pred
    gc.collect()

    print('Done!')

    return train_fold_pred, test_pred_mean


lgb_params = {
    'objective': 'binary',
    'n_estimators': 20000,
    'random_state': 42,
    'learning_rate': 8e-3,
    'subsample': 0.6,
    'subsample_freq': 1,
    'colsample_bytree': 0.4,
    'reg_alpha': 10.0,
    'reg_lambda': 1e-1,
    'min_child_weight': 256,
    'min_child_samples': 20,
    'device': 'gpu',
}

xgb_params = {'n_estimators': 10000,
              'learning_rate': 0.03689407512484644,
              'max_depth': 8,
              'colsample_bytree': 0.3723914688159835,
              'subsample': 0.780714581166012,
              'eval_metric': 'auc',
              'use_label_encoder': False,
              'gamma': 0,
              'reg_lambda': 50.0,
              'tree_method': 'gpu_hist',
              'gpu_id': 0,
              'predictor': 'gpu_predictor',
              'random_state': 42}

cat_params = {'iterations': 17298,
              'learning_rate': 0.03429054860458741,
              'reg_lambda': 0.3242286463210283,
              'subsample': 0.9433911589913944,
              'random_strength': 22.4849972385133,
              'depth': 8,
              'min_data_in_leaf': 4,
              'leaf_estimation_iterations': 8,
              'task_type': "GPU",
              'bootstrap_type': 'Poisson',
              'verbose': 500,
              'early_stopping_rounds': 200,
              'eval_metric': 'AUC'}
lgbm = LGBMClassifier(**lgb_params)

xgb = XGBClassifier(**xgb_params)

cat = CatBoostClassifier(**cat_params)
cat_train, cat_test = Stacking_Data_Loader(cat, 'cat', train, y, test, 5)
del cat
gc.collect()

lgbm_train, lgbm_test = Stacking_Data_Loader(lgbm, 'lgbm', train, y, test, 5)
del lgbm
gc.collect()

xgb_train, xgb_test = Stacking_Data_Loader(xgb, 'xgb', train, y, test, 5)
del xgb
gc.collect()

stack_x_train = np.concatenate((cat_train, lgbm_train, xgb_train), axis = 1)
stack_x_test = np.concatenate((cat_test, lgbm_test, xgb_test), axis = 1)

del cat_train, lgbm_train, xgb_train, cat_test, lgbm_test, xgb_test
gc.collect()

print(stack_x_train)

stk = StratifiedKFold(n_splits=5)

test_pred_lo = 0
fold = 1
total_auc = 0

for train_index, valid_index in stk.split(stack_x_train, y):
    x_train, y_train = stack_x_train[train_index], y[train_index]
    x_valid, y_valid = stack_x_train[valid_index], y[valid_index]

    lr = LogisticRegression(n_jobs=-1, random_state=42, C=5, max_iter=2000)
    lr.fit(x_train, y_train)

    valid_pred_lo = lr.predict_proba(x_valid)[:, 1]
    test_pred_lo += lr.predict_proba(stack_x_test)[:, 1]
    auc = roc_auc_score(y_valid, valid_pred_lo)
    total_auc += auc / 5
    print('Fold', fold, 'AUC :', auc)
    fold += 1

print('Total AUC score :', total_auc)

sub = pd.read_csv('../input/tabular-playground-series-oct-2021/sample_submission.csv')
sub['target'] = test_pred_lo
sub.to_csv('sub.csv', index = 0)
print(sub)