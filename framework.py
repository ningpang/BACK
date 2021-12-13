import os
import sys
import torch
import torch.nn as nn
from torch import optim


class framework(object):
    def __init__(self, args, train_loader, val_loader, test_loader):
        self.args = args
        self.B = args.batch_size
        self.N_for_train = args.N_for_train
        self.N = args.N
        self.K = args.K
        self.Q = args.Q
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        self.train_iter = args.train_iter
        self.val_iter = args.val_iter
        self.test_iter = args.test_iter
        self.val_step = args.val_step

        self.device = args.device


    def load_ckpt(self, ckpt):
        if os.path.isfile(ckpt):
            checkpoint = torch.load(ckpt)
            print('Load checkpoint successfully !')
            return checkpoint
        else:
            print('[ERROR] No chekpoint !')
            assert 0

    def train(self, model):
        parameters_to_optimize = filter(lambda x: x.requires_grad, model.parameters())
        optimizer = optim.SGD(parameters_to_optimize, lr=self.args.lr, weight_decay=self.args.weight_decay)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=self.args.lr_step_size)


        model = model.to(self.device)
        model.train()

        L_entropy = nn.CrossEntropyLoss()

        best_acc = 0.0
        iter_loss = 0.0
        iter_sample = 0
        for i in range(self.train_iter):
            support, query, label = next(self.train_loader)
            label = label.to(self.device)
            logits, pred, margin_loss = model(support, query, self.N_for_train, self.K, self.Q, True)
            entropy_loss = L_entropy(logits, label)
            accuracy = torch.mean((pred.view(-1) == label.view(-1)).float())
            loss = entropy_loss + margin_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            iter_loss += loss.item()
            iter_sample += 1
            msg = 'step: [{0:4}/{1:4}] | loss: {2:3.6f}, accuracy: {3:3.2f}%'
            sys.stdout.write(msg.format(i+1, self.train_iter, iter_loss/iter_sample, accuracy)+'\r')
            sys.stdout.flush()

            if (i+1)%self.val_step==0:
                with torch.no_grad():
                    acc = self.eval(model, self.N, self.K, self.Q, self.val_iter)
                    print('')
                    print('')
                    print('step: {0:4} | {1:}-way-{2:}-shot test accuracy: {3:3.2f}%'.format(i, self.N, self.K, acc * 100))

                    if acc > best_acc:
                        print('Best checkpoint')
                        if not os.path.exists(self.args.save_dict):
                            os.makedirs(self.args.save_dict)
                        save_path = os.path.join(self.args.save_dict, self.args.model + ".pth.tar")
                        torch.save({'state_dict': model.state_dict()}, save_path)
                        best_acc = acc
                    print('')

                model.train()

        print('Finish training ...')
        with torch.no_grad():
            test_acc = self.eval(model, self.N, self.K, self.Q, self.test_iter,
                                 ckpt=os.path.join(self.args.save_dict, self.args.model + '.pth.tar'))
            print("Model: {0:} | {1:}-way-{2:}-shot test |  Test accuracy: {3:3.2f}".format(self.args.model, self.N, self.K, test_acc * 100))


    def eval(self, model, N, K, Q, eval_iter, ckpt=None):
        if ckpt is None:
            data_loader = self.val_loader
        else:
            data_loader = self.test_loader
            checkpoint = self.load_ckpt(ckpt)
            model.load_state_dict(checkpoint['state_dict'], strict=False)
        model.eval()

        iter_acc = 0.0
        iter_num = 0.0
        for i in range(eval_iter):
            support, query, label = next(data_loader)
            label = label.to(self.device)
            _, pred, _ = model(support, query, N, K, Q, False)
            accuracy = torch.mean((pred.view(-1) == label.view(-1)).float())
            iter_acc += accuracy
            iter_num += 1
        return iter_acc/iter_num











