import os
import time
import torch
import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from models.DIN import DeepInterestNetwork
from utils.utils import evaluate
from data.MyDataset import SequentialRecommendationDataset

def str2bool(s):
    if s not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s == 'true'


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True)
parser.add_argument('--train_dir', required=True)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--maxlen', default=50, type=int)
parser.add_argument('--hidden_units', default=50, type=int)
parser.add_argument('--embed_dim', default=16, type=int)
parser.add_argument('--num_blocks', default=2, type=int)
parser.add_argument('--num_epochs', default=201, type=int)
parser.add_argument('--num_test_neg_item', default=100, type=int)
parser.add_argument('--num_heads', default=1, type=int)
parser.add_argument('--dropout_rate', default=0.5, type=float)
parser.add_argument('--l2_emb', default=0.0, type=float)
parser.add_argument('--device', default='cpu', type=str)
parser.add_argument('--inference_only', default=False, type=str2bool)
parser.add_argument('--state_dict_path', default=None, type=str)

args = parser.parse_args()
if not os.path.isdir(args.dataset + '_' + args.train_dir):
    os.makedirs(args.dataset + '_' + args.train_dir)
with open(os.path.join(args.dataset + '_' + args.train_dir, 'args.txt'), 'w') as f:
    f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))
f.close()

if __name__ == '__main__':
    # dataset
    dataset_train = SequentialRecommendationDataset(data_dir='data/' + args.dataset,
                                                    max_length=args.maxlen, mode='train', device=args.device)
    dataset_valid = SequentialRecommendationDataset(data_dir='data/' + args.dataset,
                                                    max_length=args.maxlen, mode='val', neg_num=args.num_test_neg_item, device=args.device)
    dataset_test = SequentialRecommendationDataset(data_dir='data/' + args.dataset,
                                                   max_length=args.maxlen, mode='test', neg_num=args.num_test_neg_item, device=args.device)

    usernum = dataset_train.user_num
    itemnum = dataset_train.item_num
    user_features_dim = dataset_train.user_features_dim
    item_features_dim = dataset_train.item_features_dim
    print('number of users: %d' % usernum, 'number of items: %d' % itemnum)

    config = {'embed_dim': args.embed_dim,
              'dim_config': {'item_id': itemnum+1, 'user_id': usernum+1,
                             'item_feature': item_features_dim, 'user_feature': user_features_dim},
              'device': args.device}
    # model = SASRec(usernum, itemnum, args).to(args.device)  # no ReLU activation in original SASRec implementation?
    model = DeepInterestNetwork(config).to(args.device)
    f = open(os.path.join(args.dataset + '_' + args.train_dir, 'log.txt'), 'w')

    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_normal_(param.data)
        except:
            pass  # just ignore those failed init layers

    # this fails embedding init 'Embedding' object has no attribute 'dim'
    # model.apply(torch.nn.init.xavier_uniform_)

    model.train()  # enable model training

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
        t_test = evaluate(model, dataset_test, usernum, itemnum, args)
        print('test (NDCG@10: %.4f, HR@10: %.4f)' % (t_test[0], t_test[1]))

    # ce_criterion = torch.nn.CrossEntropyLoss()
    # https://github.com/NVIDIA/pix2pixHD/issues/9 how could an old bug appear again...
    bce_criterion = torch.nn.BCEWithLogitsLoss()  # torch.nn.BCELoss()
    adam_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))

    T = 0.0
    t0 = time.time()

    for epoch in range(epoch_start_idx, args.num_epochs + 1):
        if args.inference_only: break  # just to decrease identition
        dataloader = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True)
        step = 0
        epoch_loss = 0.0
        for data in tqdm(dataloader):
            step += 1
            user_id, history_items, history_items_len, \
                target_item_id, neg_item_id, \
                user_features, item_features, neg_item_features = data

            # pos_logits, neg_logits = model(u, seq, pos, neg)
            pos_logits = model(target_item_id, history_items, history_items_len, user_features, item_features)
            neg_logits = model(neg_item_id, history_items, history_items_len, user_features, neg_item_features)
            pos_labels, neg_labels = torch.ones(pos_logits.shape, device=args.device), torch.zeros(neg_logits.shape,
                                                                                                   device=args.device)
            # print("\neye ball check raw_logits:"); print(pos_logits); print(neg_logits) # check pos_logits > 0, neg_logits < 0
            adam_optimizer.zero_grad()
            indices = np.where(target_item_id.cpu() != 0)
            loss = bce_criterion(pos_logits[indices], pos_labels[indices])
            loss += bce_criterion(neg_logits[indices], neg_labels[indices])
            for param in model.item_embedding.parameters():
                loss += args.l2_emb * torch.norm(param)
            loss.backward()
            adam_optimizer.step()
            epoch_loss += loss.item()
            # print("loss in epoch {} iteration {}: {}".format(epoch, step,
            #                                                  loss.item()))  # expected 0.4~0.6 after init few epochs
        print("epoch: {}, loss: {}".format(epoch, epoch_loss / step))

        if epoch % 20 == 0:
            model.eval()
            t1 = time.time() - t0
            T += t1
            print('Evaluating', end='')
            t_test = evaluate(model, dataset_test, usernum, itemnum, args)
            t_valid = evaluate(model, dataset_valid, usernum, itemnum, args)
            print('epoch:%d, time: %f(s), valid (NDCG@10: %.4f, HR@10: %.4f), test (NDCG@10: %.4f, HR@10: %.4f)'
                  % (epoch, T, t_valid[0], t_valid[1], t_test[0], t_test[1]))

            f.write(str(t_valid) + ' ' + str(t_test) + '\n')
            f.flush()
            t0 = time.time()
            model.train()

        if epoch == args.num_epochs:
            folder = args.dataset + '_' + args.train_dir
            fname = 'SASRec.epoch={}.lr={}.layer={}.head={}.hidden={}.maxlen={}.pth'
            fname = fname.format(args.num_epochs, args.lr, args.num_blocks, args.num_heads, args.hidden_units,
                                 args.maxlen)
            torch.save(model.state_dict(), os.path.join(folder, fname))

    f.close()
    print("Done")