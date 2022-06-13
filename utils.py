import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from os.path import exists
import pickle, json
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

def reduce_mem_usage(df):
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    features = df.columns
    for col in tqdm(features):
        if col == 'CUST_UID' or col == 'LABEL':
            continue
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df

def train_model(train_x, test_x, train_y, features, seed=3407):
    feature_importance_df = pd.DataFrame({'feat': features, 'imp': 0})
    KF = StratifiedKFold(n_splits=5, random_state=seed, shuffle=True)
    score_list = []
    params = {
        'objective': 'binary',
        'boosting_type': 'gbdt',
        'metric': 'auc',
        'n_jobs': -1,
        'learning_rate': 0.05,
        'num_leaves': 2 ** 6,
        'max_depth': 8,
        'tree_learner': 'serial',
        'colsample_bytree': 0.8,
        'subsample_freq': 1,
        'subsample': 0.8,
        'num_boost_round': 5000,
        'max_bin': 255,
        'verbose': -1,
        'seed': seed,
        'bagging_seed': seed,
        'feature_fraction_seed': seed,
        'early_stopping_rounds': 100,

    }
    oof_lgb = np.zeros(len(train_x))
    predictions_lgb = np.zeros((len(test_x)))

    for fold_, (trn_idx, val_idx) in enumerate(KF.split(train_x.values, train_y.values)):
        trn_data = lgb.Dataset(train_x.iloc[trn_idx][features], label=train_y.iloc[trn_idx])
        val_data = lgb.Dataset(train_x.iloc[val_idx][features], label=train_y.iloc[val_idx])
        num_round = 10000
        clf = lgb.train(
            params,
            trn_data,
            num_round,
            valid_sets=[trn_data, val_data],
            verbose_eval=100,
            early_stopping_rounds=50,
        )

        oof_lgb[val_idx] = clf.predict(train_x.iloc[val_idx][features], num_iteration=clf.best_iteration)
        predictions_lgb[:] += clf.predict(test_x[features], num_iteration=clf.best_iteration) / 5
        feature_importance_df['imp'] += clf.feature_importance() / 5
        score_list.append(roc_auc_score(train_y.iloc[val_idx], oof_lgb[val_idx]))

    print("AUC score: {}".format(roc_auc_score(train_y, oof_lgb)))
    print("F1 score: {}".format(f1_score(train_y, [1 if i >= 0.5 else 0 for i in oof_lgb])))
    print("Precision score: {}".format(precision_score(train_y, [1 if i >= 0.5 else 0 for i in oof_lgb])))
    print("Recall score: {}".format(recall_score(train_y, [1 if i >= 0.5 else 0 for i in oof_lgb])))
    print("AUC mean: {}".format(np.mean(score_list)))
    return feature_importance_df, oof_lgb, predictions_lgb

def select_features(args, train_x, test_x, train_y, feat_imp_df=None):
    off_score_dict = defaultdict(int)
    if exists(args.feat_imp_dir):
        with open(args.feat_imp_dir, 'rb') as f:
            feat_imp_df = pickle.load(f)
        # feat_imp_df = pickle.load(args.feature_imp_dir)
    elif feat_imp_df is None:
        raise FileNotFoundError
    for i in range(args.min_feature_num, args.max_feature_num):
        selected_feature = feat_imp_df.sort_values(['imp'])[-i:]['feat'].to_list()
        # You can add Multi-Seed Settings here:
        # for seed in [2021, 3407]:
        _, oof_lgb, predictions_lgb = train_model(train_x, test_x, train_y, selected_feature)
        off_score_dict[i] = roc_auc_score(train_y, oof_lgb)
    save_data(off_score_dict, args.score_dict_dir)
    return off_score_dict

def model_fusion(args, train_x, test_x, train_y):
    score_dict = np.load(args.score_dict_dir, allow_pickle=True).item()
    feat_imp_df = pd.read_pickle(args.feat_imp_dir)
    pred, oof = [], []
    for k, v in score_dict.items():
        if v > args.threshold:
            feature = feat_imp_df.sort_values(['imp'])[-k:]['feat'].to_list()
            _, oof_lgb, predictions_lgb = train_model(train_x, test_x, train_y, feature)
            pred.append(predictions_lgb)
            oof.append(oof_lgb)
    test_x['label'] = np.mean(pred, axis=0)
    test_x[['CUST_UID', 'label']].to_csv(args.saved_result_dir, index=False, header=None, sep=' ')
    print(test_x[['CUST_UID', 'label']].head())
    return test_x


def save_data(data, data_dir):
    if data_dir.endswith('.pkl'):
        data.to_pickle(data_dir)
    elif data_dir.endswith('.npy'):
        np.save(data_dir, data)