import math
import numpy as np
import torch
import torch.utils.data
import torch.nn.functional as F
from nasbench import api

INPUT = 'input'
CONV1X1 = 'conv1x1-bn-relu'
CONV3X3 = 'conv3x3-bn-relu'
MAXPOOL3X3 = 'maxpool3x3'
OUTPUT = 'output'

"""
0: sos/eos
1: no connection
2: connection
3: CONV1X1
4: CONV3X3
5: MAXPOOL3X3
6: OUTPUT
"""

MAX_EDGE = 9

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


def generate_arch(n, nasbench, need_perf=False):
    count = 0
    archs = []
    seqs = []
    valid_accs = []
    all_keys = list(nasbench.hash_iterator())
    np.random.shuffle(all_keys)
    for key in all_keys:
        fixed_stat, computed_stat = nasbench.get_metrics_from_hash(key)
        arch = api.ModelSpec(
            matrix=fixed_stat['module_adjacency'],
            ops=fixed_stat['module_operations'],
        )
        if need_perf:
            data = nasbench.query(arch)
            if data['validation_accuracy'] < 0.9:
                continue
            valid_accs.append(data['validation_accuracy'])
        archs.append(arch)
        seqs.append(convert_arch_to_seq(arch.matrix, arch.ops))
        count += 1
        if count >= n:
            return archs, seqs, valid_accs


def count_parameters(model):
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters())


class ControllerDataset(torch.utils.data.Dataset):
    def __init__(self, inputs, targets=None, train=True, sos_id=0, eos_id=0):
        super(ControllerDataset, self).__init__()
        if targets is not None:
            assert len(inputs) == len(targets)
        self.inputs = inputs
        self.targets = targets
        self.train = train
        self.sos_id = sos_id
        self.eos_id = eos_id
    
    def __getitem__(self, index):
        encoder_input = self.inputs[index]
        encoder_target = None
        if self.targets is not None:
            encoder_target = [self.targets[index]]
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


def convert_arch_to_seq(matrix, ops):
    seq = []
    n = len(matrix)
    max_n = 7
    assert n == len(ops)
    for col in range(1, max_n):
        if col >= n:
            seq += [0 for i in range(col)]
            seq.append(0)
        else:
            for row in range(col):
                seq.append(matrix[row][col]+1)
            if ops[col] == CONV1X1:
                seq.append(3)
            elif ops[col] == CONV3X3:
                seq.append(4)
            elif ops[col] == MAXPOOL3X3:
                seq.append(5)
            elif ops[col] == OUTPUT:
                seq.append(6)
    assert len(seq) == (max_n+2)*(max_n-1)/2
    return seq


def convert_seq_to_arch(seq):
    n = int(math.floor(math.sqrt((len(seq) + 1) * 2)))
    matrix = [[0 for _ in range(n)] for _ in range(n)]
    ops = [INPUT]
    for i in range(n-1):
        offset=(i+3)*i//2
        for j in range(i+1):
            matrix[j][i+1] = seq[offset+j] - 1
        if seq[offset+i+1] == 3:
            op = CONV1X1
            ops.append(op)
        elif seq[offset+i+1] == 4:
            op = CONV3X3
            ops.append(op)
        elif seq[offset+i+1] == 5:
            op = MAXPOOL3X3
            ops.append(op)
        elif seq[offset+i+1] == 6:
            op = OUTPUT
            ops.append(op)
    return matrix, ops


def move_to_cuda(tensor):
    if torch.cuda.is_available():
        return tensor.cuda()
    return tensor

