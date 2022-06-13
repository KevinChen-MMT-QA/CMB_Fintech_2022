from dataLoader import Dataset
import argparse
import warnings
from utils import train_model, save_data, select_features, model_fusion

parser = argparse.ArgumentParser()
parser.add_argument('--train_data_dir', type=str, default='./data/train.csv')
parser.add_argument('--test_data_dir', type=str, default='./data/test_A.csv')
parser.add_argument('--feat_imp_dir', type=str, default='./saved/feature_importance.pkl')
parser.add_argument('--score_dict_dir', type=str, default='./saved/score_dict.npy')
parser.add_argument('--saved_result_dir', type=str, default='./saved/result.txt')
parser.add_argument('--min_feature_num', type=int, default=10)
parser.add_argument('--max_feature_num', type=int, default=12)
parser.add_argument('--threshold', type=float, default=0.945)

args = parser.parse_known_args()[0]
warnings.filterwarnings('ignore')

if __name__ == '__main__':
    train_x, test_x, train_y, features = Dataset(args)
    feature_importance_df, oof_lgb, predictions = train_model(train_x, test_x, train_y, features)
    save_data(feature_importance_df, args.feat_imp_dir)
    score_dict = select_features(args, train_x, test_x, train_y)
    test = model_fusion(args, train_x, test_x, train_y)