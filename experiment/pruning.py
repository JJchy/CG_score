from collections import defaultdict

import numpy as np

import torch

import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from base_trainer import BaseTrainer
from utils.train import accuracy


class Pruning(BaseTrainer):
    def __init__(self):
        super().__init__()

        measure = self._load_measure()
        if self.args.prun_hard:
            measure = -1*measure

        labels = defaultdict(list)
        for i in range(len(self.train_dataset)):
            labels[int(self.train_dataset[i][1])].append(i)
        
        pruning_idx = []
        for _, idx in labels.items():
            label_measure = measure[idx]
            label_measure_rank = label_measure.argsort()
            thres_idx = int(len(label_measure)*(1-self.args.pruning))
            pruning_idx += list(np.array(idx)[label_measure_rank[:thres_idx]])

        print(len(pruning_idx))

        self.step_num = len(self.train_loader)

        self.train_dataset = torch.utils.data.Subset(self.train_dataset, pruning_idx)
        sampler = torch.utils.data.RandomSampler(self.train_dataset, replacement=True)
        self.train_loader = torch.utils.data.DataLoader(dataset=self.train_dataset,
                                                        batch_size=self.args.batch_size,
                                                        sampler=sampler,
                                                        num_workers=16,
                                                        drop_last=True)

    def _load_measure(self):
        return {
            "vi": np.load("data/vi.npz", allow_pickle=True)["arr_0"][()]['vi'],
            "c_score": -1 * np.load("data/c_score.npz", allow_pickle=True)['scores'],
            "l2norm_early": np.load(
                f"data/measures.npz", allow_pickle=True)['results'][()]["l2norm"][19, :],
            "forgetting": np.load(
                f"data/measures.npz", allow_pickle=True)['measures'][()]["forgetting"][-1, :],
        }.get(self.args.measure, np.random.rand(len(self.train_dataset)))

    def _train_single_epoch(self, epoch):
        self.model.train()

        train_total = 0
        train_correct = 0

        train_iter = iter(self.train_loader)

        for j in range(self.step_num):
            try:
                data, labels = train_iter.next()
            except:
                train_iter = iter(self.train_loader)
                data, labels = train_iter.next()       

            data = data.to(self.device)
            labels = labels.to(self.device)
            outputs = self.model(data)

            prec, = accuracy(outputs, labels, topk=(1, ))
            train_total += labels.size(0)
            train_correct += prec

            loss = self._loss(outputs, labels, epoch)

            #backprop
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        self.scheduler.step()
       
        return float(train_correct)/float(train_total)

    def _additional_parser(self, parser):
        parser.add_argument("--pruning", type=float, default=1)
        parser.add_argument("--measure", type=str, default="random")
        parser.add_argument("--prun_hard", action='store_true', default=False)

        return parser

def main():
    trainer = Pruning()
    trainer.train()

if __name__ == '__main__':
    main()
