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
from torch.hub import load_state_dict_from_url
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



parser = argparse.ArgumentParser(description='NAO CIFAR-10')

# Basic model parameters.
parser.add_argument('--data', type=str, default='./data')
parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10, cifar100, imagenet'])
parser.add_argument('--zip_file', action='store_true', default=False)
parser.add_argument('--lazy_load', action='store_true', default=False)
parser.add_argument('--output_dir', type=str, default='models')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--child_eval_batch_size', type=int, default=500)
parser.add_argument('--child_cutout_size', type=int, default=None)


parser.add_argument('--debug', action='store_true', default=False)
args = parser.parse_args()

## Ensure that child cutout size, child_eval_batch_size, dataset transforms are same as that in 


class IntResNet(ResNet):
    def __init__(self,output_layer,*args):
        self.output_layer = output_layer
        super().__init__(*args)
        
        self._layers = []
        print(list(self._modules.keys()))
        for l in list(self._modules.keys()):
            self._layers.append(l)
            if l == output_layer:
                break
        self.layers = OrderedDict(zip(self._layers,[getattr(self,l) for l in self._layers]))

    def _forward_impl(self, x):
        for l in self._layers:
            x = self.layers[l](x)

        return x

    def forward(self, x):
        return self._forward_impl(x)

def new_resnet(
    arch: str,
    outlayer: str,
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    pretrained: bool,
    progress: bool,
    **kwargs: Any
) -> IntResNet:

    '''model_urls = {
        'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
        'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
        'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
        'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
        'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
        'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
        'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
        'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
        'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
    }'''

    model = IntResNet(outlayer, block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model

def build_cifar10(model_state_dict=None, optimizer_state_dict=None, **kwargs):
    ratio = kwargs.pop('ratio')
    debug = kwargs.pop('debug')
    
    _, valid_transform = utils._data_transforms_cifar10(args.child_cutout_size)
    valid_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=valid_transform)
    
    if debug:
        indices = list(range(0,len(valid_data),len(valid_data)//100))
        valid_data = torch.utils.data.Subset(valid_data, indices)

    full_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=args.child_eval_batch_size,
        shuffle=False, pin_memory=True, num_workers=16)

    full_criterion = nn.CrossEntropyLoss(reduction='none').cuda()

    return full_queue

def generate_embeddings(full_queue, model):
    with torch.no_grad():
        all_embeddings = torch.empty(size=(0,2048))
        all_targets = torch.empty(size=(0,))
        model.eval()
        for step, (inputs, targets) in enumerate(full_queue):
            inputs = inputs.cuda()
            all_embeddings = torch.cat([all_embeddings, model(inputs).squeeze().cpu()])
            all_targets = torch.cat([all_targets, targets])
            print(all_embeddings.shape, all_targets.shape)

            
            # prec1, prec5 = utils.accuracy(logits, targets, topk=(1, 5))
            
        
    return all_embeddings, all_targets

model = new_resnet('resnet50','layer4',Bottleneck, [3, 4, 6, 3],True,True)
model = model.cuda()

print(model)
full_queue = build_cifar10(ratio=0.9, debug=args.debug)
embeddings, targets = generate_embeddings(full_queue, model)
embeddings = embeddings - torch.mean(embeddings, dim=0, keepdim=True)
embeddings = F.normalize(embeddings)
print(type(embeddings), type(targets))
torch.save(embeddings,os.path.join(args.output_dir, 'data_embeddings.{}.{}.pt'.format(args.dataset,'resnet50')))
torch.save(targets,os.path.join(args.output_dir, 'targets.{}.{}.pt'.format(args.dataset,'resnet50')))
