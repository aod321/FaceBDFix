import sys
sys.path.append("../")
import torch
import torch.nn as nn
import uuid as uid
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from utils.template import TemplateModel
from tensorboardX import SummaryWriter
from tqdm import tqdm
from TrainingBackBone.gen_data import get_loader
from TrainingBackBone.args import get_args
from TrainingBackBone.fcn import FCN_res18_FPN
from TrainRefine.icnnmodel import ICNNSegModel
from utils.calc_funcs import F1Accuracy
import os
import time

uuid = str(uid.uuid1())[0:10]
print(uuid)

args = get_args()
args.method = 'Resize'
# args.datamore = False
# print(args)

train_dataloader = get_loader(mode='train', batch_size=args.batch_size, shuffle=True, num_workers=4,
                              datamore=args.datamore)
val_dataloader = get_loader(mode='val', batch_size=1, shuffle=True, num_workers=4,
                            datamore=args.datamore)
test_dataloader = get_loader(mode='test', batch_size=1, shuffle=True, num_workers=4,
                             datamore=args.datamore)


class TrainingPipeline(TemplateModel):
    def __init__(self, argus=args):
        super(TrainingPipeline, self).__init__()
        # ============== not neccessary ===============
        self.train_logger = None
        self.eval_logger = None
        self.args = argus

        # ============== neccessary ===============
        self.writer = SummaryWriter(f'log_fpn_fcn/{uuid}')
        self.step = 0
        self.epoch = 0
        self.best_error = float('Inf')
        self.best_accu_eval = float('-Inf')
        self.best_accu_test = float('-Inf')

        self.device = torch.device("cuda:%d" % args.cuda if torch.cuda.is_available() else "cpu")
        self.train_f1_class = F1Accuracy(9)
        self.eval_f1_class = F1Accuracy(9)
        self.test_f1_class = F1Accuracy(9)

        # self.model = ICNNSegModel(in_channels=3, out_channels=2).to(self.device)
        self.model = FCN_res18_FPN(pretrained=True).to(self.device)
        if args.optim == 0:
            self.optimizer = optim.Adam(self.model.parameters(), self.args.lr1)
        elif args.optim == 1:
            self.optimizer = optim.SGD(self.model.parameters(), self.args.lr1, momentum=0.9, weight_decay=0.0)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.5)
        # self.criterion = nn.CrossEntropyLoss()
        self.criterion = nn.BCEWithLogitsLoss()
        self.metric = nn.BCEWithLogitsLoss()

        self.train_loader = train_dataloader
        self.eval_loader = val_dataloader
        self.test_loader = test_dataloader

        self.ckpt_dir = f"checkpoints/checkpoints_fcn_res18_fpn_{uuid}"
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.display_freq = args.display_freq

        # call it to check all members have been intiated
        self.check_init()

    def train_loss(self, batch):
        x, y = batch['image'].to(self.device), batch['labels'].to(self.device)
        out = self.model(x)
        loss = F.cross_entropy(out, y.argmax(dim=1, keepdim=False))
        pred_arg = out.argmax(dim=1, keepdim=False)
        self.train_f1_class.collect(pred_arg.cpu(), y.argmax(dim=1, keepdim=False))
        return loss

    def train(self):
        if self.epoch == 0:
            self.eval()
            self.test()
        self.model.train()
        self.epoch += 1
        loss = 0
        for batch in tqdm(self.train_loader):
            self.step += 1
            self.optimizer.zero_grad()

            loss = self.train_loss(batch)
            loss.backward()
            # clip_gradient(self.optimizer, grad_clip=grad_clip)
            self.optimizer.step()

            if self.step % self.display_freq == 0:
                print('epoch {}\tstep {}\tloss {:.3}'.format(self.epoch, self.step, loss.item()))
                if self.train_logger:
                    self.train_logger(self.writer)

        f1 = self.train_f1_class.calc()
        print('epoch {}\taccu {:.3}'.format(self.epoch, f1))
        self.writer.add_scalar(f'accu_train_{uuid}', f1, self.epoch)
        self.writer.add_scalar(f'loss_train_{uuid}', loss.item(), self.epoch)

    def eval_accu(self):
        loss_list = []
        for batch in tqdm(self.eval_loader):
            x, y = batch['image'].to(self.device), batch['labels'].to(self.device)
            out = self.model(x)
            # loss = self.criterion(out, y.long())
            loss = F.cross_entropy(out, y.argmax(dim=1, keepdim=False))
            pred_arg = out.argmax(dim=1, keepdim=False)
            self.eval_f1_class.collect(pred_arg, y.argmax(dim=1, keepdim=False))
            loss_list.append(loss.item())
        mean_error = np.mean(loss_list)
        F1 = self.eval_f1_class.calc()
        return F1, mean_error

    def eval(self):
        self.model.eval()
        with torch.no_grad():
            accu, mean_error = self.eval_accu()

        if accu > self.best_accu_eval:
            self.best_accu_eval = accu
            self.save_state(os.path.join(self.ckpt_dir, 'best.pth.tar'), False)
        self.save_state(os.path.join(self.ckpt_dir, '{}.pth.tar'.format(self.epoch)))
        self.writer.add_scalar(f'accu_val_{uuid}', accu, self.epoch)
        self.writer.add_scalar(f'loss_val_{uuid}', mean_error, self.epoch)
        print('epoch {}\t mean_error {:.3}\t accu {:.3}\tbest_accu_eval {:.3}'.format(self.epoch, mean_error,
                                                                                      accu, self.best_accu_eval))

    def test(self):
        self.model.eval()
        with torch.no_grad():
            accu, mean_error = self.test_accu()
        if accu > self.best_accu_test:
            self.best_accu_test = accu
        self.writer.add_scalar(f'accu_test_{uuid}', accu, self.epoch)
        self.writer.add_scalar(f'error_test_{uuid}', mean_error, self.epoch)
        print("-----------TEST-----------")
        print('epoch {}\tstep {}\terror_test {:.3}\taccu_test {:.3}\tbest_accu_test {:.3}'.format(self.epoch,
                                                                                                  self.step,
                                                                                                  mean_error,
                                                                                                  accu,
                                                                                                  self.best_accu_test))

    def test_accu(self):
        loss_list = []
        for batch in tqdm(self.test_loader):
            x, y = batch['image'].to(self.device), batch['labels'].to(self.device)
            out = self.model(x)
            loss = F.cross_entropy(out, y.argmax(dim=1, keepdim=False))
            loss_list.append(loss.item())
            pred_arg = out.argmax(dim=1, keepdim=False)
            self.test_f1_class.collect(pred_arg, y.argmax(dim=1, keepdim=False))

        mean_error = np.mean(loss_list)
        F1 = self.test_f1_class.calc()
        return F1, mean_error

    def fit(self):
        single_epoch_train_time = []
        single_epoch_val_time = []
        single_epoch_test_time = []
        for epoch in range(self.args.epochs):
            start = time.time()
            self.train()
            end = time.time()
            single_epoch_train_time.append(end - start)
            print(f"Train Time: {end - start}\n")
            self.scheduler.step()
            if (epoch + 1) % self.args.eval_per_epoch == 0:
                start = time.time()
                self.eval()
                end = time.time()
                single_epoch_val_time.append(end - start)
                print(f"Val Time: {end - start}\n")
            start = time.time()
            self.test()
            end = time.time()
            print(f"Test Time: {end - start}\n")
            single_epoch_test_time.append(end - start)
        train_time_per_epoch = np.mean(single_epoch_train_time)
        train_time_all = np.sum(single_epoch_train_time)
        val_time_per_epoch = np.mean(single_epoch_val_time)
        val_time_all = np.sum(single_epoch_val_time)
        test_time_per_epoch = np.mean(single_epoch_test_time)
        test_time_all = np.sum(single_epoch_test_time)
        print('ALL Done!!!')
        print('------------------Summary----------------------')
        print('all_train_time: {}s\tmean_train_time_per_epoch: {}s\n'
              'all_val_time:{}s\tmean_val_time_per_epoch: {}s\n'
              'all_test_time"{}s\tmean_test_time_per_epocj: {}s\n'.format(train_time_all, train_time_per_epoch,
                                                                          val_time_all, val_time_per_epoch,
                                                                          test_time_all, test_time_per_epoch)
              )


if __name__ == '__main__':
    args = get_args()
    train = TrainingPipeline(args)
    train.fit()
