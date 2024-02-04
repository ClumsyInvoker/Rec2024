import os
import time
import torch
import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import json
from copy import deepcopy

from models.PLATE import PLATE

from utils.utils import evaluate_PLATE
from data.MyDataset import PLATEDataset

def str2bool(s):
    if s not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s == 'true'


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True)
parser.add_argument('--train_dir', required=True)
parser.add_argument('--model_name', default='DSSM', type=str)
parser.add_argument('--exp_name', default='base', type=str)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--maxlen', default=50, type=int)
parser.add_argument('--embed_dim', default=16, type=int)
parser.add_argument('--num_epochs', default=201, type=int)
parser.add_argument('--num_test_neg_item', default=100, type=int)
parser.add_argument('--dropout_rate', default=0.5, type=float)
parser.add_argument('--l2_emb', default=0.0, type=float)
parser.add_argument('--device', default='cpu', type=str)
parser.add_argument('--inference_only', default=False, type=str2bool)
parser.add_argument('--state_dict_path', default=None, type=str)
parser.add_argument('--pretrain_model_path', default=None, type=str)
parser.add_argument('--save_freq', default=20, type=int)
parser.add_argument('--mode', default='full', type=str)

args = parser.parse_args()
save_dir = os.path.join(args.dataset + '_' + args.train_dir, args.exp_name)
if not os.path.isdir(args.dataset + '_' + args.train_dir):
    os.makedirs(args.dataset + '_' + args.train_dir)
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
with open(os.path.join(save_dir, 'args.txt'), 'w') as f:
    f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))
f.close()

if __name__ == '__main__':
    # dataset
    dataset_train = PLATEDataset(data_dir='data/' + args.dataset,
                                                    max_length=args.maxlen, mode='train', device=args.device)
    dataset_valid = PLATEDataset(data_dir='data/' + args.dataset,
                                                    max_length=args.maxlen, mode='val', neg_num=args.num_test_neg_item, device=args.device)
    dataset_test = PLATEDataset(data_dir='data/' + args.dataset,
                                                   max_length=args.maxlen, mode='test', neg_num=args.num_test_neg_item, device=args.device)

    usernum = dataset_train.user_num
    itemnum = dataset_train.item_num
    user_features_dim = dataset_train.user_features_dim
    item_features_dim = dataset_train.item_features_dim
    print('number of users: %d' % usernum, 'number of items: %d' % itemnum)

    config = {'embed_dim': args.embed_dim,
              'dim_config': {'item_id': itemnum+1, 'user_id': usernum+1,
                             'item_feature': item_features_dim, 'user_feature': user_features_dim},
              'device': args.device,
              'maxlen': args.maxlen}
    dataset_meta_data = json.load(open(os.path.join('data', 'dataset_meta_data.json'), 'r'))
    config['item_feature'] = dataset_meta_data[args.dataset]['item_feature']
    config['user_feature'] = dataset_meta_data[args.dataset]['user_feature']

    if args.model_name == "PLATE":
        model = PLATE(config).to(args.device)
    else:
        raise ValueError("model name not supported")
    f = open(os.path.join(save_dir, 'log.txt'), 'w')

    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_normal_(param.data)
        except:
            pass  # just ignore those failed init layers

    # this fails embedding init 'Embedding' object has no attribute 'dim'
    # model.apply(torch.nn.init.xavier_uniform_)

    model.train()  # enable model training

    # training or prompt tuning
    if args.mode == 'prompt_tuning':
        assert args.pretrain_model_path is not None
        model.load_state_dict(torch.load(args.pretrain_model_path, map_location=torch.device(args.device)))
        model.freeze_when_prompt_tunning()
        print('prompt tuning mode')
    else:
        model.freeze_when_pretrain()

    epoch_start_idx = 1
    if args.state_dict_path is not None:
        try:
            model.load_state_dict(torch.load(args.state_dict_path, map_location=torch.device(args.device)))
            tail = args.state_dict_path[args.state_dict_path.find('epoch=') + 6:]
            epoch_start_idx = int(tail[:tail.find('.')]) + 1
        except:  # in case your pytorch version is not 1.6 etc., pls debug by pdb if load weights failed
            print('failed loading state_dicts, pls check file path: ', end="")
            print(args.state_dict_path)
            print('pdb enabled for your quick check, pls type exit() if you do not need it')
            import pdb

            pdb.set_trace()

    if args.inference_only:
        model.eval()
        t_test = evaluate_PLATE(model, dataset_test, args)
        print('test (NDCG@10: %.4f, HR@10: %.4f)' % (t_test[0], t_test[1]))

    # ce_criterion = torch.nn.CrossEntropyLoss()
    # https://github.com/NVIDIA/pix2pixHD/issues/9 how could an old bug appear again...
    bce_criterion = torch.nn.BCEWithLogitsLoss()  # torch.nn.BCELoss()
    adam_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))

    T = 0.0
    t0 = time.time()
    best_val_HR = 0.0
    best_val_NDCG = 0.0
    best_HR = 0.0
    best_NDCG = 0.0
    best_epoch = -1
    best_state_dict = None

    for epoch in range(epoch_start_idx, args.num_epochs + 1):
        if args.inference_only: break  # just to decrease identition
        dataloader = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True)
        step = 0
        epoch_loss = 0.0
        train_loop = tqdm(dataloader, desc="Training Progress")
        for data in train_loop:
            step += 1
            user_id, history_items, history_items_len, target_item_id, \
                user_features, item_features, label, cold_item = data

            # logits = model(user_id, target_item_id, history_items, history_items_len, user_features, item_features)
            logits = model(user_id, target_item_id, history_items, history_items_len, user_features, item_features, cold_item)

            adam_optimizer.zero_grad()
            indices = np.where(target_item_id.cpu() != 0)
            loss = bce_criterion(logits[indices], label[indices])
            if 'item_embedding' in model.state_dict().keys():
                for param in model.item_embedding.parameters():
                    loss += args.l2_emb * torch.norm(param)
            if 'user_embedding' in model.state_dict().keys():
                for param in model.user_embedding.parameters():
                    loss += args.l2_emb * torch.norm(param)


            loss.backward()
            adam_optimizer.step()
            epoch_loss += loss.item()
            train_loop.set_description("Epoch {}/{}".format(epoch, args.num_epochs))
            train_loop.set_postfix(loss=epoch_loss/step)
            # print("loss in epoch {} iteration {}: {}".format(epoch, step,
            #                                                  loss.item()))  # expected 0.4~0.6 after init few epochs
        print("Epoch: {}, loss: {}".format(epoch, epoch_loss / step))


        if epoch % 1 == 0:
            model.eval()
            t1 = time.time() - t0
            T += t1
            print('Evaluating', end='')
            t_valid = evaluate_PLATE(model, dataset_valid, args, 'valid')
            t_test = evaluate_PLATE(model, dataset_test, args, 'test')
            print('epoch:%d, time: %f(s), valid (NDCG@10: %.4f, HR@10: %.4f), test (NDCG@10: %.4f, HR@10: %.4f)'
                  % (epoch, T, t_valid[0], t_valid[1], t_test[0], t_test[1]))

            if t_valid[1] > best_val_HR:
                best_val_HR = t_valid[1]
                best_HR = t_test[1]
                best_NDCG = t_test[0]
                best_epoch = epoch
                best_state_dict = deepcopy(model.state_dict())

            f.write(str(t_valid) + ' ' + str(t_test) + '\n')
            f.flush()
            t0 = time.time()
            model.train()

        if epoch % args.save_freq == 0 or epoch == args.num_epochs:
            folder = save_dir
            fname = 'epoch={}.lr={}.embed_dim={}.maxlen={}.l2_emb={}.pth'
            fname = fname.format(epoch, args.lr, args.embed_dim,
                                 args.maxlen, args.l2_emb)
            torch.save(model.state_dict(), os.path.join(folder, fname))

    f.write("best epoch: {}, best NDCG@10: {}, best HR@10: {}".format(best_epoch, best_NDCG, best_HR))
    f.close()
    print("best epoch: {}, best NDCG@10: {}, best HR@10: {}".format(best_epoch, best_NDCG, best_HR))
    torch.save(best_state_dict, os.path.join(save_dir, 'best.pth'))
    print("Done")