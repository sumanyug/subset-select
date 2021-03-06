import os
import sys
import glob
import time
import copy
import random
import numpy as np
import torch
import utils
import logging
import argparse
import pickle
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
from model import NASNetworkCIFAR, NASNetworkImageNet
from model_search import NASWSNetworkCIFAR, NASWSNetworkImageNet
from controller import NAO

parser = argparse.ArgumentParser(description='NAO CIFAR-10')

# Basic model parameters.
parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
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

if args.debug:
    assert(args.dataset=='cifar10')
    # Training set contains only 1000 data points
    args.child_eval_batch_size = 50
    args.child_epochs = 20
    args.child_eval_epochs = '4'
    args.controller_seed_arch = 32
    args.controller_new_arch = 16
    args.controller_epochs = 50
    assert(args.controller_expand == None)
    args.controller_batch_size = 8

utils.create_exp_dir(args.output_dir, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')


class CrossEntropyLabelSmooth(nn.Module):

    def __init__(self, num_classes, epsilon):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (-targets * log_probs).mean(0).sum()
        return loss


def get_builder(dataset):
    if dataset == 'cifar10':
        return build_cifar10
    elif dataset == 'cifar100':
        return build_cifar100
    # else:
    #     return build_imagenet
    

def build_cifar10(model_state_dict=None, optimizer_state_dict=None, **kwargs):
    epoch = kwargs.pop('epoch')
    ratio = kwargs.pop('ratio')
    debug = kwargs.pop('debug')
    
    train_transform, valid_transform = utils._data_transforms_cifar10(args.child_cutout_size)
    train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
    valid_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=valid_transform)
    
    if debug:
        indices = list(range(0,len(train_data),len(train_data)//100))
        train_data = torch.utils.data.Subset(train_data, indices)
        valid_data = torch.utils.data.Subset(valid_data, indices)
        

    num_train = len(train_data)
    assert num_train == len(valid_data)
    indices = list(range(num_train)) 
    split = int(np.floor(ratio * num_train))
    np.random.shuffle(indices)

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.child_batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
        pin_memory=True, num_workers=16)
    valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=args.child_eval_batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
        pin_memory=True, num_workers=16)
    full_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=args.child_eval_batch_size,
        shuffle=False, pin_memory=True, num_workers=16)

    
    model = NASWSNetworkCIFAR(10, args.child_layers, args.child_nodes, args.child_channels, args.child_keep_prob, args.child_drop_path_keep_prob,
                       args.child_use_aux_head, args.steps)
    model = model.cuda()
    train_criterion = nn.CrossEntropyLoss().cuda()
    eval_criterion = nn.CrossEntropyLoss().cuda()
    full_criterion = nn.CrossEntropyLoss(reduction='none').cuda()
    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    optimizer = torch.optim.SGD(
        model.parameters(),
        args.child_lr_max,
        momentum=0.9,
        weight_decay=args.child_l2_reg,
    )
    if model_state_dict is not None:
        model.load_state_dict(model_state_dict)
    if optimizer_state_dict is not None:
        optimizer.load_state_dict(optimizer_state_dict)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.child_epochs, args.child_lr_min, epoch)
    return train_queue, valid_queue, full_queue ,model, train_criterion, eval_criterion, full_criterion, optimizer, scheduler, indices[:split], indices[split:]


def build_cifar100(model_state_dict=None, optimizer_state_dict=None, **kwargs):
    epoch = kwargs.pop('epoch')
    ratio = kwargs.pop('ratio')
    train_transform, valid_transform = utils._data_transforms_cifar10(args.cutout_size)
    train_data = dset.CIFAR100(root=args.data, train=True, download=True, transform=train_transform)
    valid_data = dset.CIFAR100(root=args.data, train=True, download=True, transform=valid_transform)

    num_train = len(train_data)
    assert num_train == len(valid_data)
    indices = list(range(num_train))    
    split = int(np.floor(ratio * num_train))
    np.random.shuffle(indices)

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.child_batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
        pin_memory=True, num_workers=16)
    valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=args.child_eval_batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
        pin_memory=True, num_workers=16)
    full_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=args.child_eval_batch_size,
        shuffle=False, pin_memory=True, num_workers=16)
        
    model = NASWSNetworkCIFAR(100, args.child_layers, args.child_nodes, args.child_channels, args.child_keep_prob, args.child_drop_path_keep_prob,
                       args.child_use_aux_head, args.steps)
    model = model.cuda()
    train_criterion = nn.CrossEntropyLoss().cuda()
    eval_criterion = nn.CrossEntropyLoss().cuda()
    full_criterion = nn.CrossEntropyLoss().cuda(reduction='none')
    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    optimizer = torch.optim.SGD(
        model.parameters(),
        args.child_lr_max,
        momentum=0.9,
        weight_decay=args.child_l2_reg,
    )
    if model_state_dict is not None:
        model.load_state_dict(model_state_dict)
    if optimizer_state_dict is not None:
        optimizer.load_state_dict(optimizer_state_dict)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.child_epochs, args.child_lr_min, epoch)
    return train_queue, valid_queue, full_queue ,model, train_criterion, eval_criterion, full_criterion, optimizer, scheduler, indices[:split], indices[split:]



def child_train(train_queue, model, optimizer, global_step, arch_pool, arch_pool_prob, criterion):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.train()
    for step, (input, target) in enumerate(train_queue):
        input = input.cuda().requires_grad_()
        target = target.cuda()

        optimizer.zero_grad()
        # sample an arch to train
        arch = utils.sample_arch(arch_pool, arch_pool_prob)
        logits, aux_logits = model(input, arch, global_step)
        global_step += 1
        loss = criterion(logits, target)
        if aux_logits is not None:
            aux_loss = criterion(aux_logits, target)
            loss += 0.4 * aux_loss
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.child_grad_bound)
        optimizer.step()
        
        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        objs.update(loss.data, n)
        top1.update(prec1.data, n)
        top5.update(prec5.data, n)
        
        if (step+1) % 100 == 0:
            logging.info('Train %03d loss %e top1 %f top5 %f', step+1, objs.avg, top1.avg, top5.avg)
            logging.info('Arch: %s', ' '.join(map(str, arch[0] + arch[1])))

    return top1.avg, objs.avg, global_step


def child_valid(valid_queue, model, arch_pool, criterion):
    valid_acc_list = []
    with torch.no_grad():
        model.eval()
        for i, arch in enumerate(arch_pool):
            #for step, (inputs, targets) in enumerate(valid_queue):
            inputs, targets = next(iter(valid_queue))
            inputs = inputs.cuda()
            targets = targets.cuda()
                
            logits, _ = model(inputs, arch, bn_train=True)
            loss = criterion(logits, targets)
                
            prec1, prec5 = utils.accuracy(logits, targets, topk=(1, 5))
            valid_acc_list.append(prec1.item()/100)
            
            if (i+1) % 100 == 0:
                logging.info('Valid arch %s\n loss %.2f top1 %f top5 %f', ' '.join(map(str, arch[0] + arch[1])), loss.item(), prec1.item(), prec5.item())
        
    return valid_acc_list

def child_full_loss(full_queue, model, arch_pool, arch_pool_prob, criterion):
    # arch_subset = utils.sample_k_arch(arch_pool, , arch_pool_prob)
    arch_subset = arch_pool
    full_loss_list = []
    logging.info('Generating data for log files...')
    with torch.no_grad():
        model.eval()
        for i, arch in enumerate(arch_subset):
            arch_loss = []
            for step, (inputs, targets) in enumerate(full_queue):
                inputs = inputs.cuda()
                targets = targets.cuda()
                    
                logits, _ = model(inputs, arch, bn_train=True)
                loss = criterion(logits, targets)
                arch_loss.extend(loss.tolist())
                
                # prec1, prec5 = utils.accuracy(logits, targets, topk=(1, 5))
            
            full_loss_list.append(arch_loss)
            if i%2==0:
                logging.info('Valid arch %d: %s\n loss %.2f ', i, ' '.join(map(str, arch[0] + arch[1])), sum(arch_loss)/len(arch_loss))
        
    return arch_subset, full_loss_list


def train_and_evaluate_top_on_cifar10(archs, train_queue, valid_queue):
    res = []
    train_criterion = nn.CrossEntropyLoss().cuda()
    eval_criterion = nn.CrossEntropyLoss().cuda()
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    for i, arch in enumerate(archs):
        objs.reset()
        top1.reset()
        top5.reset()
        logging.info('Train and evaluate the {} arch'.format(i+1))
        model = NASNetworkCIFAR(args, 10, args.child_layers, args.child_nodes, args.child_channels, 0.6, 0.8,
                        True, args.steps, arch)
        model = model.cuda()
        model.train()
        optimizer = torch.optim.SGD(
            model.parameters(),
            args.child_lr_max,
            momentum=0.9,
            weight_decay=args.child_l2_reg,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 10, args.child_lr_min)
        global_step = 0
        for e in range(10):
            for step, (input, target) in enumerate(train_queue):
                input = input.cuda().requires_grad_()
                target = target.cuda()

                optimizer.zero_grad()
                # sample an arch to train
                logits, aux_logits = model(input, global_step)
                global_step += 1
                loss = train_criterion(logits, target)
                if aux_logits is not None:
                    aux_loss = train_criterion(aux_logits, target)
                    loss += 0.4 * aux_loss
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), args.child_grad_bound)
                optimizer.step()
            
                prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
                n = input.size(0)
                objs.update(loss.item(), n)
                top1.update(prec1.item(), n)
                top5.update(prec5.item(), n)
            
                if (step+1) % 100 == 0:
                    logging.info('Train epoch %03d %03d loss %e top1 %f top5 %f', e+1, step+1, objs.avg, top1.avg, top5.avg)
            scheduler.step()
        objs.reset()
        top1.reset()
        top5.reset()
        with torch.no_grad():
            model.eval()
            for step, (input, target) in enumerate(valid_queue):
                input = input.cuda()
                target = target.cuda()
            
                logits, _ = model(input)
                loss = eval_criterion(logits, target)
            
                prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
                n = input.size(0)
                objs.update(loss.item(), n)
                top1.update(prec1.item(), n)
                top5.update(prec5.item(), n)
            
                if (step+1) % 100 == 0:
                    logging.info('valid %03d %e %f %f', step+1, objs.avg, top1.avg, top5.avg)
        res.append(top1.avg)
    return res


def train_and_evaluate_top_on_cifar100(archs, train_queue, valid_queue):
    res = []
    train_criterion = nn.CrossEntropyLoss().cuda()
    eval_criterion = nn.CrossEntropyLoss().cuda()
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    for i, arch in enumerate(archs):
        objs.reset()
        top1.reset()
        top5.reset()
        logging.info('Train and evaluate the {} arch'.format(i+1))
        model = NASNetworkCIFAR(args, 100, args.child_layers, args.child_nodes, args.child_channels, 0.6, 0.8,
                        True, args.steps, arch)
        model = model.cuda()
        model.train()
        optimizer = torch.optim.SGD(
            model.parameters(),
            args.child_lr_max,
            momentum=0.9,
            weight_decay=args.child_l2_reg,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 10, args.child_lr_min)
        global_step = 0
        for e in range(10):
            for step, (input, target) in enumerate(train_queue):
                input = input.cuda().requires_grad_()
                target = target.cuda()

                optimizer.zero_grad()
                # sample an arch to train
                logits, aux_logits = model(input, global_step)
                global_step += 1
                loss = train_criterion(logits, target)
                if aux_logits is not None:
                    aux_loss = train_criterion(aux_logits, target)
                    loss += 0.4 * aux_loss
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), args.child_grad_bound)
                optimizer.step()
            
                prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
                n = input.size(0)
                objs.update(loss.data, n)
                top1.update(prec1.data, n)
                top5.update(prec5.data, n)
            
                if (step+1) % 100 == 0:
                    logging.info('Train %3d %03d loss %e top1 %f top5 %f', e+1, step+1, objs.avg, top1.avg, top5.avg)
            scheduler.step()
        objs.reset()
        top1.reset()
        top5.reset()
        with torch.no_grad():
            model.eval()
            for step, (input, target) in enumerate(valid_queue):
                input = input.cuda()
                target = target.cuda()
            
                logits, _ = model(input)
                loss = eval_criterion(logits, target)
            
                prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
                n = input.size(0)
                objs.update(loss.data, n)
                top1.update(prec1.data, n)
                top5.update(prec5.data, n)
            
                if (step+1) % 100 == 0:
                    logging.info('valid %03d %e %f %f', step+1, objs.avg, top1.avg, top5.avg)
        res.append(top1.avg)
    return res
        
        
def nao_train(train_queue, model, optimizer):
    objs = utils.AvgrageMeter()
    mse = utils.AvgrageMeter()
    nll = utils.AvgrageMeter()
    model.train()
    for step, sample in enumerate(train_queue):
        encoder_input = sample['encoder_input']
        encoder_target = sample['encoder_target']
        decoder_input = sample['decoder_input']
        decoder_target = sample['decoder_target']
        
        encoder_input = encoder_input.cuda()
        encoder_target = encoder_target.cuda().requires_grad_()
        decoder_input = decoder_input.cuda()
        decoder_target = decoder_target.cuda()
        
        optimizer.zero_grad()
        predict_value, log_prob, arch = model(encoder_input, decoder_input)
        loss_1 = F.mse_loss(predict_value.squeeze(), encoder_target.squeeze())
        loss_2 = F.nll_loss(log_prob.contiguous().view(-1, log_prob.size(-1)), decoder_target.view(-1))
        loss = args.controller_trade_off * loss_1 + (1 - args.controller_trade_off) * loss_2
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.controller_grad_bound)
        optimizer.step()
        
        n = encoder_input.size(0)
        objs.update(loss.data, n)
        mse.update(loss_1.data, n)
        nll.update(loss_2.data, n)
        
    return objs.avg, mse.avg, nll.avg


def nao_valid(queue, model):
    inputs = []
    targets = []
    predictions = []
    archs = []
    with torch.no_grad():
        model.eval()
        for step, sample in enumerate(queue):
            encoder_input = sample['encoder_input']
            encoder_target = sample['encoder_target']
            decoder_target = sample['decoder_target']
            
            encoder_input = encoder_input.cuda()
            encoder_target = encoder_target.cuda()
            decoder_target = decoder_target.cuda()
            
            predict_value, logits, arch = model(encoder_input)
            n = encoder_input.size(0)
            inputs += encoder_input.data.squeeze().tolist()
            targets += encoder_target.data.squeeze().tolist()
            predictions += predict_value.data.squeeze().tolist()
            archs += arch.data.squeeze().tolist()
    pa = utils.pairwise_accuracy(targets, predictions)
    hd = utils.hamming_distance(inputs, archs)
    return pa, hd


def nao_infer(queue, model, step, direction='+'):
    new_arch_list = []
    model.eval()
    for i, sample in enumerate(queue):
        encoder_input = sample['encoder_input']
        encoder_input = encoder_input.cuda()
        model.zero_grad()
        new_arch = model.generate_new_arch(encoder_input, step, direction=direction)
        new_arch_list.extend(new_arch.data.squeeze().tolist())
    return new_arch_list


def main():
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)
        
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    cudnn.enabled = True
    cudnn.benchmark = True
    cudnn.deterministic = True
    
    args.steps = int(np.ceil(45000 / args.child_batch_size)) * args.child_epochs

    logging.info("args = %s", args)

    if args.child_arch_pool is not None:
        logging.info('Architecture pool is provided, loading')
        with open(args.child_arch_pool) as f:
            archs = f.read().splitlines()
            archs = list(map(utils.build_dag, archs))
            child_arch_pool = archs
    elif os.path.exists(os.path.join(args.output_dir, 'arch_pool')):
        logging.info('Architecture pool is founded, loading')
        with open(os.path.join(args.output_dir, 'arch_pool')) as f:
            archs = f.read().splitlines()
            archs = list(map(utils.build_dag, archs))
            child_arch_pool = archs
    else:
        child_arch_pool = None

    child_eval_epochs = eval(args.child_eval_epochs)
    build_fn = get_builder(args.dataset)
    train_queue, valid_queue, full_queue, model, train_criterion, eval_criterion, full_criterion, optimizer, scheduler, train_indices, valid_indices = build_fn(ratio=0.6, epoch=-1, debug=args.debug)
    with open(os.path.join(args.output_dir,'train_indices.pkl'),'wb') as f:
        pickle.dump(train_indices, f)
    with open(os.path.join(args.output_dir,'valid_indices.pkl'),'wb') as f:
        pickle.dump(valid_indices, f)
    sys.exit(1)
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
    nao = nao.cuda()
    logging.info("Encoder-Predictor-Decoder param size = %fMB", utils.count_parameters_in_MB(nao))

    # Train child model
    if child_arch_pool is None:
        logging.info('Architecture pool is not provided, randomly generating now')
        child_arch_pool = utils.generate_arch(args.controller_seed_arch, args.child_nodes, 5)  # [[[conv],[reduc]]]
    child_arch_pool_prob = None

    eval_points = utils.generate_eval_points(child_eval_epochs, 0, args.child_epochs)
    step = 0
    for epoch in range(1, args.child_epochs + 1):
        lr = scheduler.get_lr()[0]
        logging.info('epoch %d lr %e', epoch, lr)
        # sample an arch to train
        train_acc, train_obj, step = child_train(train_queue, model, optimizer, step, child_arch_pool, child_arch_pool_prob, train_criterion)
        logging.info('train_acc %f', train_acc)
    
        if epoch not in eval_points:
            continue
        # Evaluate seed archs
        
        #@Samad Modify this to get validation accuracy PER data point & train accuracy PER data point
        valid_accuracy_list = child_valid(valid_queue, model, child_arch_pool, eval_criterion)

        # Output archs and evaluated error rate
        old_archs = child_arch_pool
        old_archs_perf = valid_accuracy_list

        old_archs_sorted_indices = np.argsort(old_archs_perf)[::-1]
        old_archs = [old_archs[i] for i in old_archs_sorted_indices]
        old_archs_perf = [old_archs_perf[i] for i in old_archs_sorted_indices]
        with open(os.path.join(args.output_dir, 'arch_pool.{}'.format(epoch)), 'w') as fa:
            with open(os.path.join(args.output_dir, 'arch_pool.perf.{}'.format(epoch)), 'w') as fp:
                with open(os.path.join(args.output_dir, 'arch_pool'), 'w') as fa_latest:
                    with open(os.path.join(args.output_dir, 'arch_pool.perf'), 'w') as fp_latest:
                        for arch, perf in zip(old_archs, old_archs_perf):
                            arch = ' '.join(map(str, arch[0] + arch[1]))
                            fa.write('{}\n'.format(arch))
                            fa_latest.write('{}\n'.format(arch))
                            fp.write('{}\n'.format(perf))
                            fp_latest.write('{}\n'.format(perf))
                            
        ## Save 'model' parameters as per the value
        torch.save(model.state_dict(),os.path.join(args.output_dir, 'model.{}'.format(epoch)))  
        if epoch == args.child_epochs:
            break

        # Train Encoder-Predictor-Decoder
        logging.info('Training Encoder-Predictor-Decoder')
        encoder_input = list(map(lambda x: utils.parse_arch_to_seq(x[0], 2) + utils.parse_arch_to_seq(x[1], 2), old_archs))
        # [[conv, reduc]]
        min_val = min(old_archs_perf)
        max_val = max(old_archs_perf)
        encoder_target = [(i - min_val) / (max_val - min_val) for i in old_archs_perf]

        if args.controller_expand:
            dataset = list(zip(encoder_input, encoder_target))
            n = len(dataset)
            ratio = 0.9
            split = int(n*ratio)
            np.random.shuffle(dataset)
            encoder_input, encoder_target = list(zip(*dataset))
            train_encoder_input = list(encoder_input[:split])
            train_encoder_target = list(encoder_target[:split])
            valid_encoder_input = list(encoder_input[split:])
            valid_encoder_target = list(encoder_target[split:])
            for _ in range(args.controller_expand-1):
                for src, tgt in zip(encoder_input[:split], encoder_target[:split]):
                    a = np.random.randint(0, args.child_nodes)
                    b = np.random.randint(0, args.child_nodes)
                    src = src[:4 * a] + src[4 * a + 2:4 * a + 4] + \
                            src[4 * a:4 * a + 2] + src[4 * (a + 1):20 + 4 * b] + \
                            src[20 + 4 * b + 2:20 + 4 * b + 4] + src[20 + 4 * b:20 + 4 * b + 2] + \
                            src[20 + 4 * (b + 1):]
                    train_encoder_input.append(src)
                    train_encoder_target.append(tgt)
        else:
            train_encoder_input = encoder_input
            train_encoder_target = encoder_target
            valid_encoder_input = encoder_input
            valid_encoder_target = encoder_target
        logging.info('Train data: {}\tValid data: {}'.format(len(train_encoder_input), len(valid_encoder_input)))

        nao_train_dataset = utils.NAODataset(train_encoder_input, train_encoder_target, True, swap=True if args.controller_expand is None else False)
        nao_valid_dataset = utils.NAODataset(valid_encoder_input, valid_encoder_target, False)
        nao_train_queue = torch.utils.data.DataLoader(
            nao_train_dataset, batch_size=args.controller_batch_size, shuffle=True, pin_memory=True)
        nao_valid_queue = torch.utils.data.DataLoader(
            nao_valid_dataset, batch_size=args.controller_batch_size, shuffle=False, pin_memory=True)
        nao_optimizer = torch.optim.Adam(nao.parameters(), lr=args.controller_lr, weight_decay=args.controller_l2_reg)
        for nao_epoch in range(1, args.controller_epochs+1):
            nao_loss, nao_mse, nao_ce = nao_train(nao_train_queue, nao, nao_optimizer)
            logging.info("epoch %04d train loss %.6f mse %.6f ce %.6f", nao_epoch, nao_loss, nao_mse, nao_ce)
            if nao_epoch % 100 == 0:
                pa, hs = nao_valid(nao_valid_queue, nao)
                logging.info("Evaluation on valid data")
                logging.info('epoch %04d pairwise accuracy %.6f hamming distance %.6f', epoch, pa, hs)

        # Generate new archs
        torch.save(nao.state_dict(),os.path.join(args.output_dir, 'nao.{}'.format(epoch)))

        new_archs = []
        max_step_size = 50
        predict_step_size = 0
        top100_archs = list(map(lambda x: utils.parse_arch_to_seq(x[0], 2) + utils.parse_arch_to_seq(x[1], 2), old_archs[:100]))
        nao_infer_dataset = utils.NAODataset(top100_archs, None, False)
        nao_infer_queue = torch.utils.data.DataLoader(
            nao_infer_dataset, batch_size=len(nao_infer_dataset), shuffle=False, pin_memory=True)
        while len(new_archs) < args.controller_new_arch:
            predict_step_size += 1
            logging.info('Generate new architectures with step size %d', predict_step_size)
            new_arch = nao_infer(nao_infer_queue, nao, predict_step_size, direction='+')
            for arch in new_arch:
                if arch not in encoder_input and arch not in new_archs:
                    new_archs.append(arch)
                if len(new_archs) >= args.controller_new_arch:
                    break
            logging.info('%d new archs generated now', len(new_archs))
            if predict_step_size > max_step_size:
                break
                # [[conv, reduc]]
        new_archs = list(map(lambda x: utils.parse_seq_to_arch(x, 2), new_archs))  # [[[conv],[reduc]]]
        num_new_archs = len(new_archs)
        logging.info("Generate %d new archs", num_new_archs)
        # replace bottom archs
        new_arch_pool = old_archs[:len(old_archs) - num_new_archs] + new_archs
        logging.info("Totally %d architectures now to train", len(new_arch_pool))

        child_arch_pool = new_arch_pool
        with open(os.path.join(args.output_dir, 'arch_pool'), 'w') as f:
            for arch in new_arch_pool:
                arch = ' '.join(map(str, arch[0] + arch[1]))
                f.write('{}\n'.format(arch))

        
        child_arch_pool_prob = None
        scheduler.step()


    torch.save(model.state_dict(),os.path.join(args.output_dir, 'model.final'))
    torch.save(nao.state_dict(),os.path.join(args.output_dir, 'nao.final'))
    
    topk = 3
    logging.info('Finish Searching')
    logging.info('Reranking top {} architectures'.format(topk))
    # reranking top 5
    top_archs = old_archs[:topk]
    if args.dataset == 'cifar10':
        top_archs_perf = train_and_evaluate_top_on_cifar10(top_archs, train_queue, valid_queue)
    elif args.dataset == 'cifar100':
        top_archs_perf = train_and_evaluate_top_on_cifar100(top_archs, train_queue, valid_queue)
    # else:
    #     top_archs_perf = train_and_evaluate_top_on_imagenet(top_archs, train_queue, valid_queue)
    top_archs_sorted_indices = np.argsort(top_archs_perf)[::-1]
    top_archs = [top_archs[i] for i in top_archs_sorted_indices]
    top_archs_perf = [top_archs_perf[i] for i in top_archs_sorted_indices]
    with open(os.path.join(args.output_dir, 'arch_pool.final'), 'w') as fa:
        with open(os.path.join(args.output_dir, 'arch_pool.perf.final'), 'w') as fp:
            for arch, perf in zip(top_archs, top_archs_perf):
                arch = ' '.join(map(str, arch[0] + arch[1]))
                fa.write('{}\n'.format(arch))
                fp.write('{}\n'.format(perf))
  

if __name__ == '__main__':
    main()
