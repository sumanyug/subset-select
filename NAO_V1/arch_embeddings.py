import os
import sys
import glob
import time
import copy
import random
import numpy as np
import utils
import logging
import argparse
import torch
import torch.nn as nn
from typing import Type, Any, Callable, Union, List, Dict, Optional, cast
from torch import Tensor
from collections import OrderedDict 

import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn

from torchvision.models.resnet import *
from torchvision.models.resnet import BasicBlock, Bottleneck
from torchvision.models.resnet import model_urls
from model import NASNetworkCIFAR, NASNetworkImageNet
from model_search import NASWSNetworkCIFAR, NASWSNetworkImageNet
from controller import NAO

parser = argparse.ArgumentParser(description='NAO CIFAR-10')

# Basic model parameters.
parser.add_argument('--data', type=str, default='./data')
parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10, cifar100, imagenet'])
parser.add_argument('--zip_file', action='store_true', default=False)
parser.add_argument('--lazy_load', action='store_true', default=False)
parser.add_argument('--output_dir', type=str, default='models')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--child_batch_size', type=int, default=64)
parser.add_argument('--child_eval_batch_size', type=int, default=500)
parser.add_argument('--child_epochs', type=int, default=150)
parser.add_argument('--child_layers', type=int, default=3)
parser.add_argument('--child_nodes', type=int, default=5)
parser.add_argument('--child_channels', type=int, default=20)
parser.add_argument('--child_cutout_size', type=int, default=None)
parser.add_argument('--child_grad_bound', type=float, default=5.0)
parser.add_argument('--child_lr_max', type=float, default=0.025)
parser.add_argument('--child_lr_min', type=float, default=0.001)
parser.add_argument('--child_keep_prob', type=float, default=1.0)
parser.add_argument('--child_drop_path_keep_prob', type=float, default=0.9)
parser.add_argument('--child_l2_reg', type=float, default=3e-4)
parser.add_argument('--child_use_aux_head', action='store_true', default=False)
parser.add_argument('--child_eval_epochs', type=str, default='30')
parser.add_argument('--child_arch_pool', type=str, default=None)
parser.add_argument('--child_lr', type=float, default=0.1)
parser.add_argument('--child_label_smooth', type=float, default=0.1, help='label smoothing')
parser.add_argument('--child_gamma', type=float, default=0.97, help='learning rate decay')
parser.add_argument('--child_decay_period', type=int, default=1, help='epochs between two learning rate decays')
parser.add_argument('--controller_seed_arch', type=int, default=600)
parser.add_argument('--controller_expand', type=int, default=None)
parser.add_argument('--controller_new_arch', type=int, default=300)
parser.add_argument('--controller_encoder_layers', type=int, default=1)
parser.add_argument('--controller_encoder_hidden_size', type=int, default=96)
parser.add_argument('--controller_encoder_emb_size', type=int, default=48)
parser.add_argument('--controller_mlp_layers', type=int, default=3)
parser.add_argument('--controller_mlp_hidden_size', type=int, default=200)
parser.add_argument('--controller_decoder_layers', type=int, default=1)
parser.add_argument('--controller_decoder_hidden_size', type=int, default=96)
parser.add_argument('--controller_source_length', type=int, default=40)
parser.add_argument('--controller_encoder_length', type=int, default=20)
parser.add_argument('--controller_decoder_length', type=int, default=40)
parser.add_argument('--controller_encoder_dropout', type=float, default=0)
parser.add_argument('--controller_mlp_dropout', type=float, default=0.1)
parser.add_argument('--controller_decoder_dropout', type=float, default=0)
parser.add_argument('--controller_l2_reg', type=float, default=1e-4)
parser.add_argument('--controller_encoder_vocab_size', type=int, default=12)
parser.add_argument('--controller_decoder_vocab_size', type=int, default=12)
parser.add_argument('--controller_trade_off', type=float, default=0.8)
parser.add_argument('--controller_epochs', type=int, default=1000)
parser.add_argument('--controller_batch_size', type=int, default=100)
parser.add_argument('--controller_lr', type=float, default=0.001)
parser.add_argument('--controller_optimizer', type=str, default='adam')
parser.add_argument('--controller_grad_bound', type=float, default=5.0)


parser.add_argument('--debug', action='store_true', default=False)
args = parser.parse_args()
args.steps = int(np.ceil(45000 / args.child_batch_size)) * args.child_epochs

def get_logits(full_queue, arch, model, criterion):
    with torch.no_grad():
        model.eval()
        arch_logits = torch.empty(size=(0,10))
        # arch_logits = torch.empty(size=(0,))
        for step, (inputs, targets) in enumerate(full_queue):
            inputs = inputs.cuda()
            targets = targets.cuda()
                    
            logits, _ = model(inputs, arch, bn_train=True)
            loss = criterion(logits, targets)
            arch_logits = torch.concat([arch_logits, logits.cpu()])
            # arch_losses = torch.concat([arch_losses, loss.cpu()])

        
    return  arch_logits, _



all_files = [f for f in os.listdir(args.output_dir) if os.path.isfile(os.path.join(args.output_dir, f))]

criterion = nn.CrossEntropyLoss(reduction='none').cuda()
train_transform, valid_transform = utils._data_transforms_cifar10(args.child_cutout_size)
valid_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=valid_transform)
# arch_losses = torch.empty(size=(6000, len(valid_data)))
# print("BIG LOSS")
# arch_logits = torch.empty(size=(6000, len(valid_data), 10))
# print("BIG DATA")
full_queue = torch.utils.data.DataLoader(
    valid_data, batch_size=args.child_eval_batch_size,
    shuffle=False, pin_memory=True, num_workers=16)



s = 0
nao = NAO(
    args.controller_encoder_layers,
    args.controller_encoder_vocab_size,
    args.controller_encoder_hidden_size,
    args.controller_encoder_dropout,
    args.controller_encoder_length,
    args.controller_source_length,
    args.controller_encoder_emb_size,
    args.controller_mlp_layers,
    args.controller_mlp_hidden_size,
    args.controller_mlp_dropout,
    args.controller_decoder_layers,
    args.controller_decoder_vocab_size,
    args.controller_decoder_hidden_size,
    args.controller_decoder_dropout,
    args.controller_decoder_length,
)

model = NASWSNetworkCIFAR(10, args.child_layers, args.child_nodes, args.child_channels, args.child_keep_prob, args.child_drop_path_keep_prob,
                       args.child_use_aux_head, args.steps)
model = model.cuda()
nao.load_state_dict(torch.load(os.path.join(args.output_dir,'nao.final')))
nao = nao.cuda()

arch_list = []
arch_indices = []
arch_embeddings = torch.empty(size=(6000, args.controller_encoder_emb_size*2))
assert(args.dataset=='cifar10')

curr_idx = 0

for i, f in enumerate(filter(lambda x: len(x.split('.'))==2 and x.split('.')[0]=='arch_pool' , all_files)):
    fsplit = f.split('.')
    checkpoint_num = fsplit[1]
    print(checkpoint_num)
    if not checkpoint_num.isnumeric():
        continue
    # arch_indices.append([int(checkpoint_num),int(fsplit[2])])
    model.load_state_dict(torch.load(os.path.join(args.output_dir,f'model.{int(checkpoint_num)}')))
    with open(os.path.join(args.output_dir,f)) as arch_file:
        archs_checkpoint = arch_file.read().strip('\n').split('\n')
        for j, arch in enumerate(archs_checkpoint):
            arch_list.append(arch)
            arch_indices.append([i, j])
            # arch_seq_list.append(utils.parse_arch_string_to_seq(arch,2))

            with torch.no_grad():
                seq = torch.LongTensor(utils.parse_arch_string_to_seq(arch,2)).cuda()
                # arch_embeddings = torch.concat([arch_embeddings, nao.get_arch_embedding(seq).cpu()])

                arch_embeddings[curr_idx] = nao.get_arch_embedding(seq).cpu()
                
                arch_input = utils.parse_arch_string_to_arch(arch, 2)
                arch_logits, _ = get_logits(full_queue, arch_input, model, criterion)

                torch.save(arch_logits,os.path.join(args.output_dir, 'arch_logits.{}.{}.pt'.format(args.dataset, curr_idx)))
                # torch.save(arch_logits,os.path.join(args.output_dir, 'arch_logits.{}.pt'.format(args.dataset)))
                # arch_losses[curr_idx] = losses
                # arch_logits[curr_idx] = logits
                # arch_losses = torch.concat([arch_losses, losses])
                # arch_logits = torch.concat([arch_logits , logits])
            curr_idx += 1




    # with open(os.path.join(args.output_dir, logf)) as logfile:
    #     losses = logfile.read().strip('\n')
    #     losses = torch.tensor([float(x)for j,x in enumerate(losses.split('\n'))])
    #     arch_losses = torch.concat([arch_losses, losses.unsqueeze(0)])



# nao_dataset = utils.NAODataset(arch_seq_list, None, False)
# nao_queue = torch.utils.data.DataLoader(
#     nao_dataset, batch_size=args.controller_batch_size, shuffle=False, pin_memory=True)
# for sample in nao_dataset:
#     encoder_input = sample['encoder_input']
#     encoder_input = encoder_input.cuda()
#     embs = nao.get_arch_embedding(encoder_input).cpu()
#     print(embs.shape)
#     arch_embeddings = torch.concat([arch_embeddings, embs])

arch_embeddings = arch_embeddings - torch.mean(arch_embeddings, dim=0, keepdim=True)
arch_embeddings = F.normalize(arch_embeddings)
torch.save(arch_embeddings,os.path.join(args.output_dir, 'arch_embeddings.{}.pt'.format(args.dataset)))
# torch.save(arch_losses,os.path.join(args.output_dir, 'arch_losses.{}.pt'.format(args.dataset)))
# torch.save(arch_logits,os.path.join(args.output_dir, 'arch_logits.{}.pt'.format(args.dataset)))


import pickle
with open(os.path.join(args.output_dir,'arch_list.pkl'),'wb') as f:
    pickle.dump(arch_list, f)
with open(os.path.join(args.output_dir,'arch_indices.pkl'),'wb') as f:
    pickle.dump(arch_indices, f)