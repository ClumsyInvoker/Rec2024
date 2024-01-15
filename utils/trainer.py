from utils.utils import create_dataset, Trainer, BatchLoader, EarlyStopper

import torch
import tqdm
import matplotlib.pyplot as plt

class DIENTrainer(Trainer):

    def __init__(self, model, optimizer, criterion, batch_size=None):
        super(DIENTrainer, self).__init__(model, optimizer, criterion, batch_size)

    def train(self, train_X_neg, train_y, epoch=100, trials=None, valid_X=None, valid_y=None):
        if self.batch_size:
            train_loader = BatchLoader(train_X_neg, train_y, self.batch_size)
        else:
            # 为了在 for b_x, b_y in train_loader 的时候统一
            train_loader = [[train_X_neg, train_y]]

        if trials:
            early_stopper = EarlyStopper(self.model, trials)

        train_loss_list = []
        valid_loss_list = []

        for e in tqdm(range(epoch)):
            # train part
            self.model.train()
            train_loss_ = 0
            for b_x, b_y in train_loader:
                self.optimizer.zero_grad()
                seq_len = b_x.shape[1] // 2
                pred_y, auxiliary_y = self.model(b_x[:, :seq_len + 1], b_x[:, -seq_len + 1:])

                auxiliary_true = torch.cat([torch.ones_like(auxiliary_y[0]), torch.zeros_like(auxiliary_y[1])],
                                           dim=0).view(2, -1)
                auxiliary_loss = self.criterion(auxiliary_y, auxiliary_true)
                auxiliary_loss.backward(retain_graph=True)

                train_loss = self.criterion(pred_y, b_y)
                train_loss.backward()

                self.optimizer.step()

                train_loss_ += train_loss.detach() * len(b_x)

            train_loss_list.append(train_loss_ / len(train_X_neg))

            # valid part
            if trials:
                valid_loss, valid_metric = self.test(valid_X, valid_y)
                valid_loss_list.append(valid_loss)
                if not early_stopper.is_continuable(valid_metric):
                    break

        if trials:
            self.model.load_state_dict(early_stopper.best_state)
            plt.plot(valid_loss_list, label='valid_loss')

        plt.plot(train_loss_list, label='train_loss')
        plt.legend()
        plt.show()

        print('train_loss: {:.5f} | train_metric: {:.5f}'.format(*self.test(train_X, train_y)))

        if trials:
            print('valid_loss: {:.5f} | valid_metric: {:.5f}'.format(*self.test(valid_X, valid_y)))