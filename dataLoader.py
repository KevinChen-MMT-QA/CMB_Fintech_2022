import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder

from utils import reduce_mem_usage

def Dataset(args):
    train = pd.read_csv(args.train_data_dir)
    test = pd.read_csv(args.test_data_dir)
    data = pd.concat([train, test]).reset_index(drop=True)

    features, categorical_features, continuous_features = data2feature(data)
    data = filler(data, features, continuous_features)
    data = featureCreator(data, features, continuous_features, categorical_features)
    data = reduce_mem_usage(data)

    train_x = data[~data['LABEL'].isna()].reset_index(drop=True)
    test_x = data[data['LABEL'].isna()].reset_index(drop=True)
    train_y = train_x['LABEL']
    features = [i for i in train_x.columns if i not in ['LABEL', 'CUST_UID']]
    print(train_x.shape, test_x.shape, train_y.shape)
    print('Num of Features: ', len(features))

    return train_x, test_x, train_y, features

def data2feature(data):
    features = [x for x in data.columns
                if x not in ['CUST_UID', 'LABEL']]
    categorical_features = ['MON_12_CUST_CNT_PTY_ID', 'WTHR_OPN_ONL_ICO', 'SHH_BCK',
                            'LGP_HLD_CARD_LVL', 'NB_CTC_HLD_IDV_AIO_CARD_SITU']
    continuous_features = [x for x in features if x not in categorical_features]
    return features, categorical_features, continuous_features


def filler(data, features, continuous_features):
    for f in features:
        data[f] = data[f].fillna(-2)
        data[f] = data[f].astype('str')
        data[f] = data[f].apply(lambda x: x.replace('?', '-1'))

        if f in continuous_features:
            data[f] = data[f].astype('float')
            data[f] = data[f].replace(-1, np.nan)
        else:
            lb = LabelEncoder()
            data[f] = lb.fit_transform(data[f])
        data[f] -= 2
    return data


def featureCreator(data, features, continuous_features, categorical_features):
    # Bias Features
    for group in tqdm(categorical_features):
        for f in continuous_features:
            tmp = data.groupby(group)[f].agg([sum, min, max, np.mean]).reset_index()
            tmp = pd.merge(data, tmp, on=group, how='left')
            data['{}_minus_mean_gb_{}'.format(f, group)] = data[f] - tmp['mean']
            data['{}_minus_min_gb_{}'.format(f, group)] = data[f] - tmp['min']
            data['{}_minus_max_gb_{}'.format(f, group)] = data[f] - tmp['max']
            data['{}_div_sum_gb_{}'.format(f, group)] = data[f] / tmp['sum']
    # Cross Features between Numerical Features and Categorical Features
    for i in tqdm(range(len(continuous_features))):
        for j in range(i+1, len(continuous_features)):
            for f3 in categorical_features:
                f1, f2 = continuous_features[i], continuous_features[j]
                data[f'{f1}_plus_{f2}_log_{f3}'] = (np.log1p(data[f1]) + np.log1p(data[f2])) * data[f3]
                data[f'{f1}_minus_{f2}_log_{f3}'] = (np.log1p(data[f1]) - np.log1p(data[f2])) * data[f3]
                data[f'{f1}_times_{f2}_log_{f3}'] = (np.log1p(data[f1]) * np.log1p(data[f2])) * data[f3]
                data[f'{f1}_div_{f2}_log_{f3}'] = (np.log1p(data[f1]) / np.log1p(data[f2])) * data[f3]
                data[f'{f2}_div_{f1}_log_{f3}'] = (np.log1p(data[f2]) / np.log1p(data[f1])) * data[f3]
    # Cross Features between ALL Features
    for i in tqdm(range(len(features))):
        for j in range(i+1, len(features)):
            f1, f2 = features[i], features[j]
            data[f'{f1}_plus_{f2}'] = data[f1] + data[f2]
            data[f'{f1}_minus_{f2}'] = data[f1] - data[f2]
            data[f'{f1}_times_{f2}'] = data[f1] * data[f2]
            data[f'{f1}_div_{f2}'] = data[f1] / data[f2]
            data[f'{f2}_div_{f1}'] = data[f2] / data[f1]

    print(data.shape)
    return data