from ast import Pass
from importlib.metadata import requires
import os
import sys
import glob
import pickle
import time
import copy
import random
import numpy as np
import utils
import logging
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Type, Any, Callable, Union, List, Dict, Optional, cast, Iterator
from torch import Tensor
from collections import OrderedDict 
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler, SubsetRandomSampler
import os


parser = argparse.ArgumentParser(description='NAO CIFAR-10')

parser.add_argument('--data', type=str, default='./data')
parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10, cifar100, imagenet'])
parser.add_argument('--zip_file', action='store_true', default=False)
parser.add_argument('--lazy_load', action='store_true', default=False)
parser.add_argument('--output_dir', type=str, default='models')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--ratio_data', type=float, default=0.4)
parser.add_argument('--ratio_archs', type=int, default=0.3)
parser.add_argument('--grad_bound', type=float, default=5.0)
parser.add_argument('--train', action='store_true', default=False)
parser.add_argument('--train_batch_size', type=int, default=256)
parser.add_argument('--evaluate', action='store_true', default=False)
parser.add_argument('--eval_batch_size', type=int, default=500)
parser.add_argument('--epochs', type=int, default=6)
parser.add_argument('--trialID', type=int, default=0)
parser.add_argument('--split_type', type=str, choices=['random', 'identical', 'val_only'])
parser.add_argument('--target_type', type=str, default='loss', choices=['loss', 'labels', 'logits'])


parser.add_argument('--debug', action='store_true', default=False)
args = parser.parse_args()

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')

class CrossProductInMemoryDataset(Dataset):
    def __init__(self, arch_embeddings_file, data_embeddings_file, targets_file, logit_files_root):
        self.labels = torch.load(targets_file).long()
        self.data_embeddings = torch.load(data_embeddings_file)
        self.arch_embeddings = torch.load(arch_embeddings_file)[:3000]
        self.logits_file_root = logit_files_root
        self.num_data, self.data_dim = self.data_embeddings.shape
        self.num_archs, self.arch_dim = self.arch_embeddings.shape

    def __len__(self):
        return self.num_data*self.num_archs

    def __getitem__(self, idx):
        arch_idx = int(idx) // self.num_data
        data_idx = idx % self.num_data
        logits = torch.load(f'{self.logits_file_root}.{arch_idx}.pt')
        
        return self.arch_embeddings[arch_idx], self.data_embeddings[data_idx], self.labels[data_idx], logits[data_idx]

class CustomSubsetRandomSampler(Sampler[int]):
    r"""Samples elements randomly from a given list of indices, without replacement.
    Args:
        indices (sequence): a sequence of indices
        generator (Generator): Generator used in sampling.
    """

    def __init__(self, arch_indices, num_archs, data_indices, num_data, generator=None) -> None:
        self.arch_indices = arch_indices
        self.num_archs = num_archs
        self.data_indices = data_indices
        self.num_data = num_data
        self.generator = generator

    def __iter__(self) -> Iterator[int]:
        for _ in range(len(self.arch_indices)):
            for i,j in zip(torch.randperm(len(self.data_indices), generator=self.generator), 
                            torch.randint(len(self.arch_indices),(len(self.data_indices),), generator=self.generator)):
                data_idx = self.data_indices[i]
                arch_idx = self.arch_indices[j]
                yield arch_idx*self.num_data + data_idx

    def __len__(self) -> int:
        return len(self.arch_indices)*len(self.data_indices)

random.seed(1)
np.random.seed(1)
torch.manual_seed(1)
arch_embeddings_file = 'arch_embeddings.{}.pt'.format(args.dataset)
data_embeddings_file = 'data_embeddings.{}.{}.pt'.format(args.dataset,'resnet50')
targets_file = 'targets.{}.{}.pt'.format(args.dataset,'resnet50')
logits_file = 'arch_logits.{}'.format(args.dataset)
embedding_dataset = CrossProductInMemoryDataset(*[os.path.join(args.output_dir,f) 
    for f in (arch_embeddings_file, data_embeddings_file, targets_file, logits_file)])

with open(os.path.join(args.output_dir,'train_indices.pkl'),'rb') as f:
    logits_train_indices = pickle.load(f)
with open(os.path.join(args.output_dir,'valid_indices.pkl'),'rb') as f:
    logits_val_indices = pickle.load(f)


arch_indices = torch.randperm(embedding_dataset.num_archs)
arch_val_indices = arch_indices[:int(args.ratio_archs*len(arch_indices))]
arch_train_indices = arch_indices[int(args.ratio_archs*len(arch_indices)):]


if args.split_type=='val_only':
    data_indices = copy.deepcopy(logits_val_indices)
    random.shuffle(data_indices)
    data_val_indices = data_indices[:int(args.ratio_data*len(data_indices))]
    data_train_indices = data_indices[int(args.ratio_data*len(data_indices)):]
elif args.split_type=='random':

    data_indices = [i for i in range(embedding_dataset.num_data)]
    random.shuffle(data_indices)
    data_val_indices = data_indices[:int(args.ratio_data*len(data_indices))]
    data_train_indices = data_indices[int(args.ratio_data*len(data_indices)):]
elif args.split_type=='identical':

    data_val_indices = copy.deepcopy(logits_val_indices)
    random.shuffle(data_val_indices)
    data_val_indices = data_val_indices[:int(args.ratio_data*len(data_val_indices))]

    data_train_indices = copy.deepcopy(logits_val_indices)
    random.shuffle(data_train_indices)
    data_train_indices = data_train_indices[:len(data_val_indices) - int((args.ratio_data)*len(data_val_indices))]




train_queue = torch.utils.data.DataLoader(
    embedding_dataset, batch_size=args.train_batch_size,
    sampler=CustomSubsetRandomSampler(arch_train_indices, embedding_dataset.num_archs, 
        data_train_indices, embedding_dataset.num_data),
    pin_memory=True, num_workers=0)


valid_data_queue = torch.utils.data.DataLoader(
    embedding_dataset, batch_size=args.eval_batch_size,
    sampler=CustomSubsetRandomSampler(arch_train_indices, embedding_dataset.num_archs, 
        data_val_indices, embedding_dataset.num_data),  
    pin_memory=True, num_workers=0)

valid_arch_queue = torch.utils.data.DataLoader(
    embedding_dataset, batch_size=args.eval_batch_size,
    sampler=CustomSubsetRandomSampler(arch_val_indices, embedding_dataset.num_archs, 
        data_train_indices, embedding_dataset.num_data),  
    pin_memory=True, num_workers=0)

valid_data_arch_queue = torch.utils.data.DataLoader(
    embedding_dataset, batch_size=args.eval_batch_size,
    sampler=CustomSubsetRandomSampler(arch_val_indices, embedding_dataset.num_archs, 
        data_val_indices, embedding_dataset.num_data),  
    pin_memory=True, num_workers=0)

queue_dict = {}
queue_dict['Train'] = train_queue
queue_dict['Valid Data'] = valid_data_queue
queue_dict['Valid Arch'] = valid_arch_queue
queue_dict['Valid Data + Arch'] = valid_data_arch_queue

# full_queue = torch.utils.data.DataLoader(
#     embedding_dataset, batch_size=args.eval_batch_size,
#     pin_memory=True, num_workers=8)


class ModelApproximator(nn.Module):
    def __init__(self, arch_in_features, data_in_features):
        super(ModelApproximator, self).__init__()
        self.arch_in_features = arch_in_features
        self.data_in_features = data_in_features

        self.model = nn.Sequential(
            nn.Linear(arch_in_features+data_in_features, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512,128),
            nn.ReLU(inplace=True),
            nn.Linear(128,10),
            )
    def forward(self, x):
        logits = self.model(x)
        return logits

class ModelApproximator2(nn.Module):
    def __init__(self, arch_in_features, data_in_features):
        super(ModelApproximator2, self).__init__()
        self.arch_in_features = arch_in_features
        self.data_in_features = data_in_features

        self.model = nn.Sequential(
            nn.Linear(arch_in_features+data_in_features, 256),
            nn.Dropout(p=0.3),
            nn.ReLU(inplace=True),
            nn.Linear(256,10),
            )

    # x represents our data
    def forward(self, x):
        logits = self.model(x)
        return logits


def validate(model_approximator, loss_fn, criterion_logits, criterion_loss, queue, max_batches = None):
    t0 = time.time()
    with torch.no_grad():
        loss_tracker = utils.AvgrageMeter()
        accuracy_tracker = utils.AvgrageMeter()
        ce_tracker = utils.AvgrageMeter()
        loss_tracker.reset()
        model_approximator.eval()
        for i, (arch_embeddings, data_embeddings, labels, target_logits) in enumerate(queue):
            input = torch.concat([arch_embeddings, data_embeddings],1).cuda()
            labels = labels.cuda()
            target_logits = target_logits.cuda()
            target_prob = F.softmax(target_logits, dim=1)
            logits = model_approximator(input)

            prec1 = utils.accuracy(logits, labels, topk=(1,))[0]


            pred_loss = loss_fn(logits, labels)
            target_loss = loss_fn(target_logits, labels)

            ce = criterion_logits(logits, target_prob)
            loss_diff = criterion_loss(pred_loss, target_loss)
            loss_tracker.update(loss_diff.item(),1)
            accuracy_tracker.update(prec1.data, 1)
            ce_tracker.update(ce.item(), 1)
            # if i%100==99:
            #     logging.info(f'{time.time()-t0:.2f}:  Batch {i+1}/{len(queue)} Avg loss {loss_tracker.avg:.3f} Acc {accuracy_tracker.avg:.2f}')

            if max_batches and i==max_batches-1:
                break
        # logging.info(f'{time.time()-t0:.2f}:  Batch {i+1}/{len(queue)} Avg loss {loss_tracker.avg:.3f} Acc {accuracy_tracker.avg:.2f}')
        return  loss_tracker.avg, accuracy_tracker.avg, ce_tracker.avg

model_approximator = ModelApproximator2(embedding_dataset.arch_dim, embedding_dataset.data_dim).cuda()
import time
t0 = time.time()
loss_fn = nn.CrossEntropyLoss(reduction='none').cuda()
criterion_loss = nn.MSELoss().cuda()
criterion_logits = nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.AdamW(model_approximator.parameters(), lr=0.001, betas=(0.9, 0.999), 
        eps=1e-08, weight_decay=0.005, amsgrad=False)
if args.train:
    for epoch in range(args.epochs):
        loss_tracker = utils.AvgrageMeter()
        loss_tracker.reset()
        model_approximator.train()
        for i, (arch_embeddings, data_embeddings, labels, target_logits) in enumerate(train_queue):
            optimizer.zero_grad()
            input = torch.concat([arch_embeddings, data_embeddings],1).cuda()
            labels = labels.cuda()
            target_logits = target_logits.cuda()
            target_prob = F.softmax(target_logits, dim=1)
            logits = model_approximator(input)
            pred_loss = loss_fn(logits, labels)
            if args.target_type=='loss':
                target_loss = loss_fn(target_logits, labels)
                approximator_loss = criterion_loss(pred_loss, target_loss)
            elif args.target_type=='labels':
                approximator_loss = torch.mean(pred_loss)
            elif args.target_type=='logits':
                approximator_loss = criterion_logits(logits, target_prob)
            approximator_loss.backward()
            loss_tracker.update(approximator_loss.item(),1)
            nn.utils.clip_grad_norm_(model_approximator.parameters(), args.grad_bound)
            optimizer.step()
            if i==1 or i%200==199:
                logging.info(f'{time.time()-t0:.2f}: Train epoch {epoch+1}:{i+1}/{len(train_queue)} loss {loss_tracker.avg:.3f}')
                loss_tracker.reset()
                for queue_name, queue in queue_dict.items():
                    avg_loss_diff, avg_acc, avg_ce = validate(model_approximator, loss_fn, criterion_logits, criterion_loss, queue, max_batches=75)
                    logging.info('{}: Loss Diff {} | Acc {} | ce {}\n'.format(queue_name, avg_loss_diff, avg_acc, avg_ce))
            if approximator_loss.item()==0:
                break
                
        torch.save(model_approximator.state_dict(),os.path.join(args.output_dir, 'model_approximator.{}.{}'.format(args.trialID, epoch)))
else:
    model_approximator.load_state_dict(torch.load(os.path.join(args.output_dir,'model_approximator.{}.{}'.format(args.trialID, max(0, args.epochs-1)))))    

results_dict = {}

if args.evaluate:
    for queue_name, queue in queue_dict.items():
        logging.info(f'{queue_name} Loss Evaluation starting...')
        avg_loss_diff, avg_acc, avg_ce = validate(model_approximator, loss_fn, criterion_logits, criterion_loss, queue, max_batches=1000)
        results_dict[queue_name] = {'loss_diff':avg_loss_diff, 'acc': avg_acc, 'cross_entropy':avg_ce}

    with open(os.path.join(args.output_dir,'approximator_results.{}.txt'.format(args.trialID)),'w') as f:
        for queue_name, res in results_dict.items():
            f.write('{}: Loss Diff {} | Acc {} | CE {}\n'.format(queue_name, res['loss_diff'], res['acc'], res['cross_entropy']))


# loss_vector = embedding_dataset.losses
# var_loss_data, mean_loss_data = torch.var_mean(loss_vector[arch_train_indices][:,data_train_indices], dim=0)
# print(torch.mean(mean_loss_data), torch.mean(var_loss_data))
# var_loss_data, mean_loss_data = torch.var_mean(loss_vector[arch_val_indices][:,data_train_indices], dim=0)
# print(torch.mean(mean_loss_data), torch.mean(var_loss_data))
# var_loss_data, mean_loss_data = torch.var_mean(loss_vector[arch_train_indices][:,data_val_indices], dim=0)
# print(torch.mean(mean_loss_data), torch.mean(var_loss_data))
# var_loss_data, mean_loss_data = torch.var_mean(loss_vector[arch_val_indices][:,data_val_indices], dim=0)
# print(torch.mean(mean_loss_data), torch.mean(var_loss_data))


# var_loss_data, mean_loss_data = torch.var_mean(loss_vector[arch_train_indices][:,data_train_indices], dim=1)
# print(torch.mean(mean_loss_data), torch.mean(var_loss_data))
# var_loss_data, mean_loss_data = torch.var_mean(loss_vector[arch_val_indices][:,data_train_indices], dim=1)
# print(torch.mean(mean_loss_data), torch.mean(var_loss_data))
# var_loss_data, mean_loss_data = torch.var_mean(loss_vector[arch_train_indices][:,data_val_indices], dim=1)
# print(torch.mean(mean_loss_data), torch.mean(var_loss_data))
# var_loss_data, mean_loss_data = torch.var_mean(loss_vector[arch_val_indices][:,data_val_indices], dim=1)
# print(torch.mean(mean_loss_data), torch.mean(var_loss_data))
