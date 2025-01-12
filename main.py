import os
import yaml

import numpy as np

# torch
import torch
import torch.nn as nn
import torch.optim as optim

# record
from tqdm import tqdm
from tensorboardX import SummaryWriter

from util.parser import get_parser
from util.util import import_class, accuracy, AverageMeter

from net import BYOL
from optim.lars import LARS, exclude_bias_and_norm

class Processor():

    def __init__(self, arg):
        self.arg = arg
        self.save_arg()
        self.load_data()
        self.load_model()
        self.load_optimizer()
        self.load_scheduler()
        self.load_state()
        self.best_train_acc1 = 0
        self.best_test_acc1 = 0

    def load_data(self):
        Feeder = import_class(self.arg.feeder)
        self.trainloader = torch.utils.data.DataLoader(
            dataset=Feeder(**self.arg.train_feeder_args),
            batch_size=self.arg.batch_size,
            shuffle=True,
            num_workers=self.arg.num_worker,
            pin_memory=True)
        if self.arg.phase == 'test':
            self.testloader = torch.utils.data.DataLoader(
                dataset=Feeder(**self.arg.test_feeder_args),
                batch_size=self.arg.batch_size,
                shuffle=True,
                num_workers=self.arg.num_worker,
                pin_memory=True)
            
    def load_model(self):
        self.device = self.arg.device
        if self.arg.phase == 'train':
            self.model = BYOL(self.arg).to(self.device)
        elif self.arg.phase == 'test':
            Model = import_class(arg.model)
            self.model = Model(**arg.model_args).to(self.device)
            self.loss = nn.CrossEntropyLoss().to(self.device)

    def load_optimizer(self):
        if self.arg.optimizer == 'SGD':
            self.optimizer = optim.SGD(self.model.parameters(),lr=self.arg.base_lr,momentum=0.9,nesterov=self.arg.nesterov,weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer == 'Adam':
            self.optimizer = optim.Adam(self.model.parameters(),lr=self.arg.base_lr,weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer == 'LARS':
            self.optimizer = LARS(self.model.parameters(), lr=self.arg.base_lr, weight_decay=arg.weight_decay,weight_decay_filter=exclude_bias_and_norm,lars_adaptation_filter=exclude_bias_and_norm)
        else:
            raise ValueError()

    def load_scheduler(self):
        self.scheduler = optim.lr_scheduler.MultiplicativeLR(self.optimizer,lr_lambda = lambda epoch: 0.95)

    def load_state(self):
        # automatically resume from checkpoint if it exists
        path = os.path.join(self.arg.work_dir,'checkpoint.pth')
        if os.path.exists(path):
            ckpt = torch.load(path, map_location='cpu', weights_only=True)
            self.model.load_state_dict(ckpt['model'], strict=False)
            self.optimizer.load_state_dict(ckpt['optimizer'])
            self.scheduler.load_state_dict(ckpt['scheduler'])
            self.arg.start_epoch = ckpt['epoch']
        else:
            self.arg.start_epoch = 0
        # load pretrain
        if self.arg.phase == 'test':
            state_dict = torch.load(self.arg.pretrained, map_location='cpu')
            missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
        #    assert missing_keys == ['fc.weight', 'fc.bias'] # and unexpected_keys == []
            self.model.fc.weight.data.normal_(mean=0.0, std=0.01)
            self.model.fc.bias.data.zero_()
            if self.arg.weights == 'freeze':
                self.model.requires_grad_(False)
                self.model.fc.requires_grad_(True)
            classifier_parameters, model_parameters = [], []
            for name, param in self.model.named_parameters():
                if name in {'fc.weight', 'fc.bias'}:
                    classifier_parameters.append(param)
                else:
                    model_parameters.append(param)

            param_groups = [dict(params=classifier_parameters, lr=self.arg.base_lr)]
            if self.arg.weights == 'finetune':
                param_groups.append(dict(params=model_parameters, lr=self.arg.base_lr))

    def save_arg(self):
        # save arg
        arg_dict = vars(self.arg)
        if not os.path.exists(self.arg.work_dir):
            os.makedirs(self.arg.work_dir)
        with open('{}/config.yaml'.format(self.arg.work_dir), 'w') as f:
            yaml.dump(arg_dict, f)

    def train(self):
        train_writer = SummaryWriter(self.arg.work_dir)
        loss_min=float('inf')
        self.model.train()
        for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
            print("Epoch:[{}/{}]".format(epoch+1,self.arg.num_epoch))
            
            loss_epoch = 0
            
            for batch_idx, [data1, data2] in enumerate(self.trainloader):

                # get data
                data1 = data1.float().to(self.device, non_blocking=True)
                data2 = data2.float().to(self.device, non_blocking=True)

                # forward
                loss = self.model.forward(data1,data2)

                # backward
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                # record
                n_batch = len(self.trainloader)
                loss_epoch += loss.item()

                step = epoch * n_batch + batch_idx
                train_writer.add_scalar('loss_step', loss.data.item(), step)
                train_writer.add_scalar('lr', self.scheduler.get_last_lr(), step)

            self.scheduler.step()

            print("Mean loss: {} LR: {}".format(loss_epoch/n_batch, self.scheduler.get_last_lr()))
            
            state = dict(epoch=epoch + 1, model=self.model.encoder.state_dict(),optimizer=self.optimizer.state_dict(),scheduler=self.scheduler.state_dict())
            torch.save(state, os.path.join(self.arg.work_dir,'checkpoint.pth'))
            if loss_epoch<loss_min :
                loss_min=loss_epoch
                torch.save(state, os.path.join(self.arg.work_dir,'best.pth'))

            if epoch%10 == 0 :
                model_path = '{}/epoch{}_model.pt'.format(self.arg.work_dir,epoch)
                torch.save(self.model.state_dict(),model_path)

    def eval(self, save_score=False):
        train_writer = SummaryWriter(self.arg.work_dir)
        self.model.eval()
        score_frag = []
        
        top1_train = AverageMeter('Acc@1')
        top1_test = AverageMeter('Acc@1')

        if self.arg.weights == 'finetune':
            self.model.train()
        elif self.arg.weights == 'freeze':
            self.model.eval()
        else:
            assert False
        for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
            print("Epoch:[{}/{}]".format(epoch+1,self.arg.num_epoch))
            # train
            loss_epoch = 0
            for batch_idx, (data, label) in enumerate(self.trainloader):
                # get data
                data = data.float().to(self.device, non_blocking=True)
                label = label.long().to(self.device, non_blocking=True)
                # forward
                output = self.model.forward(data)
                loss = self.loss(output,label)
                # backward
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                # record
                n_batch = len(self.trainloader)
                loss_epoch += loss.item()
                
                acc1 = accuracy(output, label, topk=(1, ))
                top1_train.update(acc1[0].item(), data.size(0))
                
                step = epoch * n_batch + batch_idx
                train_writer.add_scalar('loss_step', loss.data.item(), step)
                train_writer.add_scalar('lr', self.scheduler.get_last_lr(), step)

            self.scheduler.step()
            print("Train Acc@1: {}".format(top1_train.avg))
            # test
            with torch.no_grad():
                for batch_idx, (data, label) in enumerate(self.trainloader):
                    data = data.float().to(self.device, non_blocking=True)
                    label = label.long().to(self.device, non_blocking=True)
                    
                    output = self.model(data)
                    loss = self.loss(output, label)

                    acc1 = accuracy(output, label, topk=(1,))
                    top1_test.update(acc1[0].item(), data.size(0))
                    score_frag.append(output.data.cpu().numpy())
            
                print("Eval Acc@1: {}".format(top1_test.avg))

                score = np.concatenate(score_frag)
        
                if top1_test.avg > self.best_test_acc1:
                    bestout=score
                    self.best_test_acc1 = top1_test.avg

        print("Eval Best Acc@1: {}".format(self.best_test_acc1))
        
        if save_score:
            np.save('{}/score.npy'.format(self.arg.work_dir),bestout)

    def start(self):
        if self.arg.phase == 'train':
            print('Parameters:\n{}\n'.format(str(vars(self.arg))))
            self.train()

        elif self.arg.phase == 'test':
            if self.arg.pretrained is None:
                raise ValueError('Please appoint --pretrain.')
            self.eval(save_score=self.arg.save_score)

if __name__ == '__main__':
    parser = get_parser()

    # load arg form config file
    p = parser.parse_args()
    if p.config is not None:
        with open(p.config, 'r') as f:
            default_arg = yaml.load(f,Loader=yaml.FullLoader)
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG: {}'.format(k))
                assert (k in key)
        parser.set_defaults(**default_arg)

    arg = parser.parse_args()
    processor = Processor(arg)
    processor.start()
