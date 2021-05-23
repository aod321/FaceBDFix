import sys

sys.path.append("..")
import os
import time
import torch
import torch.nn as nn
import torchvision
import numpy as np
from tqdm import tqdm
import torch.optim as optim
from datetime import datetime
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from utils.template import TemplateModel
from utils.calc_funcs import F1Accuracy
from TrainRefine.args import get_args
from TrainRefine.data_factory import get_loader
from TrainRefine.model import RefineModel
from TrainingBackBone.fcn import FCN_res18_FPN
from utils.calc_funcs import affine_mapback
from torch.distributions.categorical import Categorical


args = get_args()
args.method = 'Resize'
print(args)
# args.datamore = False
# print(args)

train_dataloader = get_loader(mode='train')
val_dataloader = get_loader(mode='val')
test_dataloader = get_loader(mode='test')


class TrainingPipeline(TemplateModel):
    def __init__(self, argus=args, fixed=True):
        super(TrainingPipeline, self).__init__()
        # ============== not neccessary ===============
        self.train_logger = None
        self.eval_logger = None
        self.args = argus
        self.fixed = fixed

        # ============== neccessary ===============
        TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())
        self.writer = SummaryWriter(
            f'logs_refine/{TIMESTAMP}_{self.uuid}')
        self.step = 0
        self.epoch = 0
        self.best_error = float('Inf')
        self.best_accu = float('-Inf')
        self.best_accu_test = float('-Inf')

        self.device = torch.device("cuda:%d" % self.args.cuda if torch.cuda.is_available() else "cpu")

        self.model = RefineModel().to(self.device)
        if args.optim == 0:
            self.optimizer = optim.Adam([
                {'params': self.model.parameters(), 'lr': self.args.lr}
            ])
        elif args.optim == 1:
            self.optimizer = optim.SGD([
                {'params': self.model.parameters(), 'lr': self.args.lr, 'momentum': 0.9, 'weight_deacy': 0.0}
            ])
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.5)
        self.criterion = nn.CrossEntropyLoss()
        self.metric = nn.CrossEntropyLoss()

        self.train_loader = train_dataloader
        self.eval_loader = val_dataloader
        self.test_loader = test_dataloader
        self.ckpt_dir = f"checkpoints/refine{args.lr}" \
                        f"optim{args.optim}_{self.uuid}"
        self.display_freq = args.display_freq
        self.f1_class_train = F1Accuracy(2)
        self.f1_class_eval = F1Accuracy(2)
        self.f1_class_test = F1Accuracy(2)

        # call it to check all members have been intiated
        self.check_init()

    def save_state(self, fname, model, optim=True):
        state = {}
        if isinstance(model, torch.nn.DataParallel):
            state['model'] = model.module.state_dict()
        else:
            state['model'] = model.state_dict()

        if optim:
            state['optimizer'] = self.optimizer.state_dict()
        state['step'] = self.step
        state['epoch'] = self.epoch
        torch.save(state, fname)
        print('save model at {}'.format(fname))

    def load_state(self, fname, model, optim=True, map_location=None):
        state = torch.load(fname, map_location=map_location)

        if isinstance(self.model, torch.nn.DataParallel):
            model.module.load_state_dict(state['model'])

        else:
            model.load_state_dict(state['model'])

        if optim and 'optimizer' in state:
            self.optimizer.load_state_dict(state['optimizer'])
        self.step = state['step']
        self.epoch = state['epoch']
        self.best_accu = state['best_accu']
        # self.best_test = state['best_test_accu']
        print('load model from {}'.format(fname))

    def forward_body(self, batch, f1_class, mode='train'):
        image, label = batch['image'].to(self.device), \
                       batch['labels'].to(self.device)
        rough = batch['rough_mask'].to(self.device)
        label[label > 0] = 1
        N, C, H, W = image.shape
        assert label.shape == (N, 1, H, W), print(label.shape)
        assert rough.shape == label.shape, print(rough.shape, label.shape)
        # ---------  Patches Segmentation---------
        pred_patches = self.model(image, pred=rough)
        loss_patches = self.criterion(pred_patches, label)
        pred_patches_arg = (torch.sigmoid(pred_patches) > 0.5).long()
        f1_class.collect(pred_patches_arg, label)
        return loss_patches

    def train_loss(self, batch):
        loss = self.forward_body(batch, f1_class=self.f1_class_train)
        return loss

    def train(self):
        self.model.train()
        self.epoch += 1
        epoch_loss_list = []
        for batch in tqdm(self.train_loader):
            self.step += 1
            self.optimizer.zero_grad()
            loss = self.train_loss(batch)
            loss.backward()
            epoch_loss_list.append(loss.item())
            # clip_gradient(self.optimizer, grad_clip=grad_clip)
            self.optimizer.step()
            if self.step % self.display_freq == 0:
                print('epoch {}\tstep {}\tloss {:.3}'.format(self.epoch, self.step, loss.item()))

        f1 = self.f1_class_train.calc()
        print(f'epoch {self.epoch}\taccu {f1}\t')
        self.writer.add_scalars('train_accu',
                                {
                                    'accu': f1
                                },
                                global_step=self.epoch)
        self.writer.add_scalar(f't_loss_train', np.mean(epoch_loss_list), self.epoch)

    def eval_body(self, eval_func, best_dict, mode='val'):
        os.makedirs(os.path.join(self.ckpt_dir, mode), exist_ok=True)
        self.model.eval()

        with torch.no_grad():
            f1, mean_loss = eval_func()

        if mean_loss < self.best_error:
            self.best_error = mean_loss
            self.save_state(fname=os.path.join(self.ckpt_dir, mode, 'best_error.pth.tar'),
                            model=self.model,
                            optim=False)

        if f1 > best_dict['f1']:
            best_dict['f1'] = f1
            self.save_state(fname=os.path.join(self.ckpt_dir, mode, 'best_f1.pth.tar'),
                            model=self.model,
                            optim=False)
        self.save_state(os.path.join(self.ckpt_dir, mode, f"{self.epoch}_f1:{f1}.pth.tar"),
                        model=self.model)

        self.writer.add_scalars(f'{mode}_accu',
                                {
                                    'accu': f1
                                },
                                global_step=self.epoch)
        self.writer.add_scalar(f'{mode}_mean_loss', mean_loss, self.epoch)

        log_txt = f"uuid {self.uuid} \t epoch {self.epoch}\t {mode}_mean_loss {mean_loss}\t \n" \
                  f"{mode}_f1 {f1}\t\n"
        log_txt = log_txt + f"Best Accu: {best_dict['f1']}\n" + "----------------------\n"
        print(log_txt)
        self.save_to_file(os.path.join(self.ckpt_dir, mode, f'log_{self.epoch}.txt'), log_txt)
        # torch.cuda.empty_cache()

    def eval_accu(self):
        epoch_loss_list = []
        for batch in tqdm(self.eval_loader):
            loss = self.forward_body(batch, f1_class=self.f1_class_eval, mode='val')
            epoch_loss_list.append(loss.item())
        # ---------------------end for batch-----------------------------------------
        mean_error = np.mean(epoch_loss_list)
        f1 = self.f1_class_eval.calc()

        return f1, mean_error

    def test_accu(self):
        epoch_loss_list = []
        for batch in tqdm(self.test_loader):
            loss = self.forward_body(batch, f1_class=self.f1_class_test, mode='test')
            epoch_loss_list.append(loss.item())
        # ---------------------end for batch-----------------------------------------
        mean_error = np.mean(epoch_loss_list)
        f1 = self.f1_class_test.calc()
        return f1, mean_error

    def eval(self):
        best_dict = {
            'f1': self.best_accu
        }
        self.eval_body(eval_func=self.eval_accu, best_dict=best_dict, mode='val')

    def test(self):
        best_dict = {
            'f1': self.best_accu_test
        }
        self.eval_body(eval_func=self.test_accu, best_dict=best_dict, mode='test')

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
        self.save_to_file(os.path.join(self.ckpt_dir, "time.txt"),
                          'all_train_time: {}s\tmean_train_time_per_epoch: {}s\n'
                          'all_val_time:{}s\tmean_val_time_per_epoch: {}s\n'
                          'all_test_time"{}s\tmean_test_time_per_epocj: {}s\n'.format(
                              train_time_all, train_time_per_epoch,
                              val_time_all, val_time_per_epoch,
                              test_time_all, test_time_per_epoch)
                          )

    @staticmethod
    def save_to_file(file_name, contents):
        fh = open(file_name, 'w')
        fh.write(contents)
        fh.close()


if __name__ == '__main__':
    # import sys
    # sys.path.append("../")

    args = get_args()

    # train_dataloader = get_loader(mode='train', batch_size=args.batch_size, shuffle=True, num_workers=4)
    # val_dataloader = get_loader(mode='val', batch_size=args.batch_size, shuffle=True, num_workers=4)

    train = TrainingPipeline(args)

    train.fit()
