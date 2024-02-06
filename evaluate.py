import os
import time
import torch
import argparse
from torch.utils.data import DataLoader
from torch.nn import functional as F
from tqdm import tqdm
import numpy as np
import json
from copy import deepcopy

from utils.utils import evaluate_by_model_name
from utils.create_model import create_model
from utils.create_dataset import create_dataset


def str2bool(s):
    if s not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s == 'true'


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True)
parser.add_argument('--test_dir', required=True)
parser.add_argument('--state_dict_path', required=True, type=str)
parser.add_argument('--model_name', default='DSSM_PTCR', type=str)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--maxlen', default=50, type=int)
parser.add_argument('--embed_dim', default=16, type=int)
parser.add_argument('--num_test_neg_item', default=100, type=int)
parser.add_argument('--device', default='cpu', type=str)
parser.add_argument('--inference_only', default=False, type=str2bool)
parser.add_argument('--top_K', default=10, type=int)
parser.add_argument('--ablation_setting', default='original', type=str)

args = parser.parse_args()
save_dir = os.path.join(args.dataset + '_' + args.test_dir)
if not os.path.isdir(args.dataset + '_' + args.test_dir):
    os.makedirs(args.dataset + '_' + args.test_dir)
# with open(os.path.join(save_dir, 'args.txt'), 'a') as f:
#     f.write(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + '\n')
#     f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))
# f.close()

if __name__ == '__main__':
    # dataset
    dataset_test = create_dataset('data/' + args.dataset, args, 'test')

    dataset_meta_data = json.load(open(os.path.join('data', 'dataset_meta_data.json'), 'r'))

    usernum = dataset_meta_data[args.dataset]['user_num']
    itemnum = dataset_meta_data[args.dataset]['item_num']
    user_features_dim = dataset_meta_data[args.dataset]['user_feature']['dim']
    item_features_dim = dataset_meta_data[args.dataset]['item_feature']['dim']
    print('number of users: %d' % usernum, 'number of items: %d' % itemnum)

    config = {'embed_dim': args.embed_dim,
              'dim_config': {'item_id': itemnum + 1, 'user_id': usernum + 1,
                             'item_feature': item_features_dim, 'user_feature': user_features_dim},
              'prompt_embed_dim': args.embed_dim,
              'prompt_net_hidden_size': args.embed_dim,
              'device': args.device,
              'maxlen': args.maxlen,
              'prompt_ablation_setting': args.ablation_setting
              }

    model = create_model(args.model_name, args, config)

    f = open(os.path.join(save_dir, args.model_name + '_log.txt'), 'a')
    f.write(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + ' model: ' + args.model_name + '\n')

    model.load_state_dict(torch.load(args.state_dict_path, map_location=torch.device(args.device)))

    model.eval()
    t_test = evaluate_by_model_name(args.model_name, model, dataset_test, args, 'test')
    print('test (NDCG@%d: %.4f, HR@%d: %.4f)' % (args.top_K, t_test[0], args.top_K, t_test[1]))
    f.write('test (NDCG@%d: %.4f, HR@%d: %.4f)' % (args.top_K, t_test[0], args.top_K, t_test[1]))

    print("Done")
