from pathlib import Path
import datetime
import os
import json
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

import utils.args
import utils.train
from model.resnet import ResNet18, ResNet50

class BaseTrainer:
    def __init__(self):
        self._pre_setting()
        self._load_data()
        self._get_model()
        self._get_optimizer_and_scheduler()
        self._get_logger()

    def _additional_parser(self, parser):
        return parser

    def _pre_setting(self):
        self.execute_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
        self.args = utils.args.get_args(self._additional_parser)
        print(self.args)

        # Set device and Fix seed
        self.device = 'cuda' if (torch.cuda.is_available() and not self.args.no_cuda) else 'cpu'
        utils.train.fix_seed(self.args.seed)

        self.max_test_acc = 0
        self.max_epoch = 0

    def _load_data(self):
        if self.args.dataset == "CIFAR10":
            CIFAR10_train_transf = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2434, 0.2615)),
                ])
    
            CIFAR10_test_transf = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2434, 0.2615)),
                ])
    
            self.train_dataset = torchvision.datasets.CIFAR10(root=f'./dataset/CIFAR10', 
                                                            train=True,
                                                            transform=CIFAR10_train_transf,
                                                            download=True)
            self.test_dataset = torchvision.datasets.CIFAR10(root=f'./dataset/CIFAR10', 
                                                            train=False,
                                                            transform=CIFAR10_test_transf,
                                                            download=True)
            
            self.input_size = [32, 32, 3]
            self.label_size = 10  
                                                        
        elif self.args.dataset == "CIFAR100":
            # https://github.com/weiaicunzai/pytorch-cifar100/
            train_mean = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
            train_std = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(train_mean, train_std),
                ])
    
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(train_mean, train_std),
                ])
    
            
            self.train_dataset = torchvision.datasets.CIFAR100(root=f'./dataset/CIFAR100', 
                                                        train=True,
                                                        transform=transform_train,
                                                        download=True)
            self.test_dataset = torchvision.datasets.CIFAR100(root=f'./dataset/CIFAR100', 
                                                        train=False,
                                                        transform=transform_test,
                                                        download=True)  
      
            self.input_size = [32, 32, 3]
            self.label_size = 100
        
        elif self.args.dataset == "FMNIST":
            FMNIST_train_transf = transforms.Compose([
                transforms.RandomCrop(28, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
                ])
    
            FMNIST_test_transf = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
                ])
    
            self.train_dataset = torchvision.datasets.FashionMNIST(root=f'./dataset/FMNIST', 
                                                            train=True,
                                                            transform=FMNIST_train_transf,
                                                            download=True)
            self.test_dataset = torchvision.datasets.FashionMNIST(root=f'./dataset/FMNIST', 
                                                            train=False,
                                                            transform=FMNIST_test_transf,
                                                            download=True)
            self.eval_dataset = torchvision.datasets.FashionMNIST(root=f'./dataset/FMNIST', 
                                                            train=True,
                                                            transform=FMNIST_test_transf,
                                                            download=True)
            
            self.input_size = [28, 28, 1]
            self.label_size = 10

        else:
            raise NotImplementedError
                                                        
        self.train_loader = torch.utils.data.DataLoader(dataset=self.train_dataset,
                                                        batch_size=self.args.batch_size,
                                                        shuffle=True,
                                                        num_workers=16,
                                                        pin_memory=False)
    
        self.test_loader = torch.utils.data.DataLoader(dataset=self.test_dataset,
                                                        batch_size=self.args.batch_size,
                                                        shuffle=False,
                                                        num_workers=16,
                                                        pin_memory=False)

    def _get_model(self):
        if self.args.model == "resnet18":
            model = ResNet18(in_channels=self.input_size[-1], 
                             num_classes=self.label_size,
                             first_conv_stride=first_conv_stride).to(self.device)
        elif self.args.model == "resnet50":
            model = ResNet50(in_channels=self.input_size[-1], 
                             num_classes=self.label_size,
                             first_conv_stride=first_conv_stride).to(self.device)
        else:
            raise NotImplementedError

        self.model = model

    def _get_optimizer_and_scheduler(self):
        self.CE_loss = nn.CrossEntropyLoss()
        model_params = self.model.parameters()

        # optimizer
        if self.args.optim == "SGD":
            optimizer = optim.SGD(
                model_params, 
                lr=lr, 
                weight_decay=self.args.regularizer
            )
        elif self.args.optim == "Momentum":
            optimizer = optim.SGD(
                model_params,
                lr=lr, 
                weight_decay=self.args.regularizer, 
                momentum=0.9
            )
        elif self.args.optim == "ADAM":
            optimizer = optim.Adam(
                model_params, 
                lr=lr, 
                weight_decay=self.args.regularizer
            )
        else:
            raise NotImplementedError

        #scheduler
        if self.args.scheduler == "CosineTorch":
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=10,
                eta_min=0
            )
        elif self.args.scheduler == "Cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, 
                T_max=self.args.epochs
            )
        elif self.args.scheduler == "Step":
            scheduler = optim.lr_scheduler.StepLR(
                optimizer, 60, gamma = 0.2
            )
        elif self.args.scheduler == "MultiStep":
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer, [60, 120, 160], gamma = 0.2
            )
        else:
            raise NotImplementedError

        self.optimizer = optimizer
        self.scheduler = scheduler
    
    def _get_logger(self):
        self.log_dir = os.path.join(
            "test" if self.args.test_exp else "runs", 
            self.execute_time,
            self.args.dataset, 
            str(self.args.epochs)
        )
        if self.args.exp_name is not None:
            self.log_dir = os.path.join(self.log_dir, self.args.exp_name)

        Path(self.log_dir).mkdir(parents=True, exist_ok=True)

        with open(os.path.join(self.log_dir, 'args.txt'), 'w') as f:
            json.dump(vars(self.args), f, indent=4) 

        # figures
        self.fig_dir = os.path.join(self.log_dir, "figures")
        Path(self.fig_dir).mkdir(parents=True, exist_ok=True)
        
        # models
        self.models_dir = os.path.join(self.log_dir, "models")
        Path(self.models_dir).mkdir(parents=True, exist_ok=True)
        
        # tensorboard
        self.writer = SummaryWriter(log_dir=self.log_dir)

    def _loss(self, logit, label, index=None, epoch=None):
        return self.CE_loss(logit, label)

    def _train_single_epoch(self, epoch):
        self.model.train()

        train_total = 0
        train_correct = 0

        for j, (data, labels) in enumerate(self.train_loader):
            data = data.to(self.device)
            labels = labels.to(self.device)
            outputs = self.model(data)

            prec, = utils.train.accuracy(outputs, labels, topk=(1, ))
            train_total += labels.size(0)
            train_correct += prec

            loss = self._loss(outputs, labels, epoch)

            #backprop
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        self.scheduler.step()
       
        return float(train_correct)/float(train_total)
    
    def _evaluate_single_epoch(self):
        self.model.eval()
        
        correct = 0
        total = 0

        with torch.no_grad():
            for data, labels in self.test_loader:
                data = data.to(self.device)
                labels = labels.to(self.device)
                logits = self.model(data)
                prec, = utils.train.accuracy(logits, labels, topk=(1, ))
                total += labels.size(0)
                correct += prec
                        
        acc = float(correct) / float(total)
        return acc
    
    def _print_and_write(self, epoch, acc_list, mode='train', t_epoch=None):
        prt_str = f'Epoch [{epoch}/{self.args.epochs}] {mode} accuracy: {acc_list:.2f}%'
        if t_epoch is not None:
            elapsed_time = time.time()- t_epoch
            prt_str += f', time: {elapsed_time:.2f}s'
        print(prt_str)
        
        acc = acc_list
        self.writer.add_scalar(f"Accuracy/{mode}", acc, epoch)
        if mode == 'test':
            self.curr_test_acc = acc
            self.curr_epoch = epoch
            if acc > self.max_test_acc:
                self.max_test_acc = acc
                self.max_epoch = epoch

    def _train_summary(self):
        print(f"max/min:{self.max_test_acc:.2f}%({self.max_epoch})/{self.curr_test_acc:.2f}")
        print(self.args)

    def save_model(self, epoch):
        torch.save(
            self.models[0].state_dict(), 
            os.path.join(self.models_dir, "model_" + str(epoch) + ".pt")
        )

    def train(self):
        # epoch starts from 1!
        t_training = time.time()

        for epoch in range(1, self.args.epochs+1):
            t_epoch = time.time()
            # train accuracy
            train_acc_list = self._train_single_epoch(epoch)
            self._print_and_write(epoch, train_acc_list, t_epoch=t_epoch)

            # test accuracy
            test_acc_list = self._evaluate_single_epoch()
            self._print_and_write(epoch, test_acc_list, mode='test')

            if self.args.model_save:
                self.save_model(epoch)

            # record time
            self.writer.add_scalar("Time", time.time()-t_epoch, epoch)
        
        print("Total Learning time: {:2f}s".format(time.time() - t_training))
        self._train_summary()

def main():
    trainer = BaseTrainer()
    trainer.train()

if __name__ == '__main__':
    main()
