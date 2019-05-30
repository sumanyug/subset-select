import os
import numpy as np
import logging
import torch
import shutil
import torch
import torch.nn as nn
import torch.utils.data
import torchvision.transforms as transforms

B=5

class AvgrageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt
      

def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0/batch_size))
    return res


class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img
    
    
def _data_transforms_cifar10(cutout_size):
    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    if cutout_size is not None:
        train_transform.transforms.append(Cutout(cutout_size))

    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])
    return train_transform, valid_transform


class NAODataset(torch.utils.data.Dataset):
    def __init__(self, inputs, targets=None, train=True, sos_id=0, eos_id=0, swap=True):
        super(NAODataset, self).__init__()
        if targets is not None:
            assert len(inputs) == len(targets)
        self.inputs = inputs
        self.targets = targets
        self.train = train
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.swap = swap
    
    def __getitem__(self, index):
        encoder_input = self.inputs[index]
        encoder_target = None
        if self.targets is not None:
            encoder_target = [self.targets[index]]
        if self.swap:
            a = np.random.randint(0, 5)
            b = np.random.randint(0, 5)
            encoder_input = encoder_input[:4 * a] + encoder_input[4 * a + 2:4 * a + 4] + \
                            encoder_input[4 * a:4 * a + 2] + encoder_input[4 * (a + 1):20 + 4 * b] + \
                            encoder_input[20 + 4 * b + 2:20 + 4 * b + 4] + encoder_input[20 + 4 * b:20 + 4 * b + 2] + \
                            encoder_input[20 + 4 * (b + 1):]
        if self.train:
            decoder_input = [self.sos_id] + encoder_input[:-1]
            sample = {
                'encoder_input': torch.LongTensor(encoder_input),
                'encoder_target': torch.FloatTensor(encoder_target),
                'decoder_input': torch.LongTensor(decoder_input),
                'decoder_target': torch.LongTensor(encoder_input),
            }
        else:
            sample = {
                'encoder_input': torch.LongTensor(encoder_input),
                'decoder_target': torch.LongTensor(encoder_input),
            }
            if encoder_target is not None:
                sample['encoder_target'] = torch.FloatTensor(encoder_target)
        return sample
    
    def __len__(self):
        return len(self.inputs)


def count_parameters_in_MB(model):
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)/1e6


def save_checkpoint(state, is_best, save):
    filename = os.path.join(save, 'checkpoint.pth.tar')
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(save, 'model_best.pth.tar')
        shutil.copyfile(filename, best_filename)
      

def save(model_path, args, model, epoch, step, optimizer):
    state_dict = {
        'args': args,
        'model': model.state_dict() if model else {},
        'epoch': epoch,
        'step': step,
        'optimizer': optimizer.state_dict()
    }
    filename = os.path.join(model_path, 'checkpoint{}.pt'.format(epoch))
    newest_filename = os.path.join(model_path, 'checkpoint.pt')
    torch.save(state_dict, filename)
    shutil.copyfile(filename, newest_filename)
  

def load(model_path):
    newest_filename = os.path.join(model_path, 'checkpoint.pt')
    if not os.path.exists(newest_filename):
        return None, None, 0, 0, None
    state_dict = torch.load(newest_filename)
    args = state_dict['args']
    model_state_dict = state_dict['model']
    epoch = state_dict['epoch']
    step = state_dict['step']
    optimizer_state_dict = state_dict['optimizer']
    return args, model_state_dict, epoch, step, optimizer_state_dict

  
def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.makedirs(path)
        print('Experiment dir : {}'.format(path))

    if scripts_to_save is not None:
        os.makedirs(os.path.join(path, 'scripts'), exist_ok=True)
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)


def sample_arch(arch_pool, prob=None):
    N = len(arch_pool)
    indices = [i for i in range(N)]
    if prob is not None:
        prob = np.array(prob, dtype=np.float32)
        prob = prob / prob.sum()
        index = np.random.choice(indices, p=prob)
    else:
        index = np.random.choice(indices)
    arch = arch_pool[index]
    return arch


def generate_arch(n, num_nodes, num_ops=7):
    def _get_arch():
        arch = []
        for i in range(2, num_nodes+2):
            p1 = np.random.randint(0, i)
            op1 = np.random.randint(0, num_ops)
            p2 = np.random.randint(0, i)
            op2 = np.random.randint(0 ,num_ops)
            arch.extend([p1, op1, p2, op2])
        return arch
    archs = [[_get_arch(), _get_arch()] for i in range(n)] #[[[conv],[reduc]]]
    return archs


def build_dag(arch):
    if arch is None:
        return None, None
    # assume arch is the format [idex, op ...] where index is in [0, 5] and op in [0, 10]
    arch = list(map(int, arch.strip().split()))
    length = len(arch)
    conv_dag = arch[:length//2]
    reduc_dag = arch[length//2:]
    return conv_dag, reduc_dag


def parse_arch_to_seq(cell, branch_length):
    assert branch_length in [2, 3]
    seq = []
    
    def _parse_op(op):
        if op == 0:
            return 7, 12
        if op == 1:
            return 8, 11
        if op == 2:
            return 8, 12
        if op == 3:
            return 9, 11
        if op == 4:
            return 10, 11

    for i in range(B):
        prev_node1 = cell[4*i]+1
        prev_node2 = cell[4*i+2]+1
        if branch_length == 2:
            op1 = cell[4*i+1] + 7
            op2 = cell[4*i+3] + 7
            seq.extend([prev_node1, op1, prev_node2, op2])
        else:
            op11, op12 = _parse_op(cell[4*i+1])
            op21, op22 = _parse_op(cell[4*i+3])
            seq.extend([prev_node1, op11, op12, prev_node2, op21, op22]) #nopknopk
    return seq


def parse_seq_to_arch(seq, branch_length):
    n = len(seq)
    assert branch_length in [2, 3]
    assert n // 2 // 5 // 2 == branch_length
    
    def _parse_cell(cell_seq):
        cell_arch = []
        
        def _recover_op(op1, op2):
            if op1 == 7:
                return 0
            if op1 == 8:
                if op2 == 11:
                    return 1
                if op2 == 12:
                    return 2
            if op1 == 9:
                return 3
            if op1 == 10:
                return 4
        if branch_length == 2:
            for i in range(B):
                p1 = cell_seq[4*i] - 1
                op1 = cell_seq[4*i+1] - 7
                p2 = cell_seq[4*i+2] - 1
                op2 = cell_seq[4*i+3] - 7
                cell_arch.extend([p1, op1, p2, op2])
            return cell_arch
        else:
            for i in range(B):
                p1 = cell_seq[6*i] - 1
                op11 = cell_seq[6*i+1]
                op12 = cell_seq[6*i+2]
                op1 = _recover_op(op11, op12)
                p2 = cell_seq[6*i+3] - 1
                op21 = cell_seq[6*i+4]
                op22 = cell_seq[6*i+5]
                op2 = _recover_op(op21, op22)
                cell_arch.extend([p1, op1, p2, op2])
            return cell_arch
    conv_seq = seq[:n//2]
    reduc_seq = seq[n//2:]
    conv_arch = _parse_cell(conv_seq)
    reduc_arch = _parse_cell(reduc_seq)
    arch = [conv_arch, reduc_arch]
    return arch


def pairwise_accuracy(la, lb):
    n = len(la)
    assert n == len(lb)
    total = 0
    count = 0
    for i in range(n):
        for j in range(i+1, n):
            if la[i] >= la[j] and lb[i] >= lb[j]:
                count += 1
            if la[i] < la[j] and lb[i] < lb[j]:
                count += 1
            total += 1
    return float(count) / total


def hamming_distance(la, lb):
    N = len(la)
    assert N == len(lb)
  
    def _hamming_distance(s1, s2):
        n = len(s1)
        assert n == len(s2)
        c = 0
        for i, j in zip(s1, s2):
            if i != j:
                c += 1
        return c
  
    dis = 0
    for i in range(N):
        line1 = la[i]
        line2 = lb[i]
        dis += _hamming_distance(line1, line2)
    return dis / N


def generate_eval_points(eval_epochs, stand_alone_epoch, total_epochs):
    if isinstance(eval_epochs, list):
        return eval_epochs
    assert isinstance(eval_epochs, int)
    res = []
    eval_point = eval_epochs - stand_alone_epoch
    while eval_point + stand_alone_epoch <= total_epochs:
        res.append(eval_point)
        eval_point += eval_epochs
    return res