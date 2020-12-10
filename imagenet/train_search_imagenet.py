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
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
from model_ws import NASNet
from nao_controller import NAO

parser = argparse.ArgumentParser(description='SemiNAS Search')

# Basic model parameters.
parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
parser.add_argument('--data', type=str, default='data')
parser.add_argument('--zip_file', action='store_true', default=False)
parser.add_argument('--lazy_load', action='store_true', default=False)
parser.add_argument('--output_dir', type=str, default='models')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--eval_batch_size', type=int, default=512)
parser.add_argument('--layers', type=int, default=21)
parser.add_argument('--dropout', type=float, default=0)
parser.add_argument('--model_init', type=str, default='he_fout', choices=['he_fin', 'he_fout'])
parser.add_argument('--no_decay_keys', type=str, default='bn', choices=['None', 'bn', 'bn#bias'])
parser.add_argument('--max_num_updates', type=int, default=1000)
parser.add_argument('--grad_clip', type=float, default=0)
parser.add_argument('--weight_decay', type=float, default=4e-5)
parser.add_argument('--arch_pool', type=str, default=None)
parser.add_argument('--lr', type=float, default=0.4)
parser.add_argument('--lr_scheduler', type=str, default='cosine', choices=['cosine', 'linear'])
parser.add_argument('--label_smooth', type=float, default=0.1, help='label smoothing')
parser.add_argument('--gamma', type=float, default=0.97, help='learning rate decay')
parser.add_argument('--decay_period', type=int, default=1, help='epochs between two learning rate decays')
parser.add_argument('--num_workers', type=int, default=32)
parser.add_argument('--log_interval', type=int, default=100)

parser.add_argument('--width_stages', type=str, default='24,40,80,96,192,320')
parser.add_argument('--n_cell_stages', type=str, default='4,4,4,4,4,1')
parser.add_argument('--stride_stages', type=str, default='2,2,2,1,2,1')

parser.add_argument('--controller_iterations', type=int, default=3)
parser.add_argument('--controller_seed_arch', type=int, default=100)
parser.add_argument('--controller_new_arch', type=int, default=100)
parser.add_argument('--controller_random_arch', type=int, default=4000)
parser.add_argument('--controller_up_sample_ratio', type=int, default=None)
parser.add_argument('--controller_encoder_layers', type=int, default=1)
parser.add_argument('--controller_decoder_layers', type=int, default=1)
parser.add_argument('--controller_mlp_layers', type=int, default=2)
parser.add_argument('--controller_hidden_size', type=int, default=16)
parser.add_argument('--controller_mlp_hidden_size', type=int, default=64)
parser.add_argument('--controller_dropout', type=float, default=0.1)
parser.add_argument('--controller_l2_reg', type=float, default=1e-4)
parser.add_argument('--controller_trade_off', type=float, default=0.8)
parser.add_argument('--controller_pretrain_epochs', type=int, default=10000)
parser.add_argument('--controller_epochs', type=int, default=1000)
parser.add_argument('--controller_batch_size', type=int, default=100)
parser.add_argument('--controller_lr', type=float, default=0.001)
parser.add_argument('--controller_optimizer', type=str, default='adam')
parser.add_argument('--controller_grad_clip', type=float, default=5.0)
args = parser.parse_args()

utils.create_exp_dir(args.output_dir, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')


def build_imagenet(model_state_dict, optimizer_state_dict, **kwargs):
    valid_ratio = kwargs.pop('valid_ratio', None)
    valid_num = kwargs.pop('valid_num', None)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(
            brightness=0.4,
            contrast=0.4,
            saturation=0.4,
            hue=0.1),
        transforms.ToTensor(),
        normalize,
    ])
    valid_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    if args.zip_file:
        logging.info('Loading data from zip file')
        traindir = os.path.join(args.data, 'train.zip')
        if args.lazy_load:
            data = utils.ZipDataset(traindir)
        else:
            logging.info('Loading data into memory')
            data = utils.InMemoryZipDataset(traindir, num_workers=args.num_workers)
    else:
        logging.info('Loading data from directory')
        traindir = os.path.join(args.data, 'train')
        if args.lazy_load:
            data = dset.ImageFolder(traindir)
        else:
            logging.info('Loading data into memory')
            data = utils.InMemoryDataset(traindir, num_workers=args.num_workers)
       
    num_data = len(data)
    indices = list(range(num_data))
    np.random.shuffle(indices)
    if valid_ratio is not None:
        split = int(np.floor(1 - valid_ratio * num_data))
        train_indices = sorted(indices[:split])
        valid_indices = sorted(indices[split:])
    else:
        assert valid_num is not None
        train_indices = sorted(indices[valid_num:])
        valid_indices = sorted(indices[:valid_num])

    train_data = utils.WrappedDataset(data, train_indices, train_transform)
    valid_data = utils.WrappedDataset(data, valid_indices, valid_transform)
    logging.info('train set = %d', len(train_data))
    logging.info('valid set = %d', len(valid_data))

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(train_indices),
        pin_memory=True, num_workers=args.num_workers, drop_last=False)
    valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=args.eval_batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(valid_indices),
        pin_memory=True, num_workers=args.num_workers, drop_last=False)
    
    model = NASNet(args.width_stages, args.n_cell_stages, args.stride_stages, args.dropout)
    model.init_model(args.model_init)
    model.set_bn_param(0.1, 0.001)
    logging.info("param size = %d", utils.count_parameters(model))

    if model_state_dict is not None:
        model.load_state_dict(model_state_dict)

    if torch.cuda.device_count() > 1:
        logging.info("Use %d %s", torch.cuda.device_count(), "GPUs !")
        model = nn.DataParallel(model)
    model = model.cuda()

    if args.no_decay_keys:
        keys = args.no_decay_keys.split('#')
        net_params=[model.module.get_parameters(keys, mode='exclude'),
                    model.module.get_parameters(keys, mode='include')]
        optimizer = torch.optim.SGD([
            {'params': net_params[0], 'weight_decay': args.weight_decay},
            {'params': net_params[1], 'weight_decay': 0},],
            args.lr,
            momentum=0.9,
            nesterov=True,
        )
    else:
        optimizer = torch.optim.SGD(
            model.parameters(),
            args.lr,
            momentum=0.9,
            weight_decay=args.weight_decay,
            nesterov=True,
        )
    if optimizer_state_dict is not None:
        optimizer.load_state_dict(optimizer_state_dict)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.decay_period, gamma=args.gamma)

    train_criterion = utils.CrossEntropyLabelSmooth(1000, args.label_smooth).cuda()
    eval_criterion = nn.CrossEntropyLoss().cuda()
    return train_queue, valid_queue, model, train_criterion, eval_criterion, optimizer, scheduler


def child_train(train_queue, model, optimizer, global_step, arch_pool, arch_pool_prob, criterion, log_interval=100):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.train()
    for step, (input, target) in enumerate(train_queue):
        input = utils.move_to_cuda(input)
        target = utils.move_to_cuda(target)

        optimizer.zero_grad()
        # sample an arch to train
        arch = utils.sample_arch(arch_pool, arch_pool_prob)
        logits = model(input, arch)
        loss = criterion(logits, target)
        loss.backward()
        optimizer.step()
        
        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        objs.update(loss.data, n)
        top1.update(prec1.data, n)
        top5.update(prec5.data, n)

        global_step += 1
        
        if global_step % log_interval == 0:
            logging.info('Train %03d loss %e top1 %f top5 %f', global_step, objs.avg, top1.avg, top5.avg)
            logging.info('Arch: %s', ' '.join(map(str, arch)))
        
        if global_step >= args.max_num_updates:
            break

    return top1.avg, objs.avg, global_step


def child_valid(valid_queue, model, arch_pool, criterion, log_interval=1):
    valid_acc_list = []
    with torch.no_grad():
        model.eval()
        for i, arch in enumerate(arch_pool):
            # for step, (input, target) in enumerate(valid_queue):
            inputs, targets = next(iter(valid_queue))
            inputs = utils.move_to_cuda(inputs)
            targets = utils.move_to_cuda(targets)
                
            logits = model(inputs, arch, bn_train=True)
            loss = criterion(logits, targets)
                
            prec1, prec5 = utils.accuracy(logits, targets, topk=(1, 5))
            valid_acc_list.append(prec1.data/100)
            
            if (i+1) % log_interval == 0:
                logging.info('Valid arch %s\n loss %.2f top1 %f top5 %f', ' '.join(map(str, arch)), loss, prec1, prec5)
        
    return valid_acc_list


def controller_train(train_queue, model, optimizer):
    objs = utils.AvgrageMeter()
    mse = utils.AvgrageMeter()
    nll = utils.AvgrageMeter()
    model.train()
    for step, sample in enumerate(train_queue):
        encoder_input = utils.move_to_cuda(sample['encoder_input'])
        encoder_target = utils.move_to_cuda(sample['encoder_target'])
        decoder_input = utils.move_to_cuda(sample['decoder_input'])
        decoder_target = utils.move_to_cuda(sample['decoder_target'])
        
        optimizer.zero_grad()
        predict_value, log_prob, arch = model(encoder_input, decoder_input)
        loss_1 = F.mse_loss(predict_value.squeeze(), encoder_target.squeeze())
        loss_2 = F.nll_loss(log_prob.contiguous().view(-1, log_prob.size(-1)), decoder_target.view(-1))
        loss = args.controller_trade_off * loss_1 + (1 - args.controller_trade_off) * loss_2
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.controller_grad_clip)
        optimizer.step()
        
        n = encoder_input.size(0)
        objs.update(loss.data, n)
        mse.update(loss_1.data, n)
        nll.update(loss_2.data, n)
        
    return objs.avg, mse.avg, nll.avg


def controller_infer(queue, model, step, direction='+'):
    new_arch_list = []
    model.eval()
    for i, sample in enumerate(queue):
        encoder_input = utils.move_to_cuda(sample['encoder_input'])
        model.zero_grad()
        new_arch = model.generate_new_arch(encoder_input, step, direction=direction)
        new_arch_list.extend(new_arch.data.squeeze().tolist())
    return new_arch_list


def train_controller(model, train_input, train_target, epochs):
    logging.info('Train data: {}'.format(len(train_input)))
    dataset = utils.NAODataset(train_input, train_target, True)
    queue = torch.utils.data.DataLoader(
        dataset, batch_size=args.controller_batch_size, shuffle=True, pin_memory=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.controller_lr, weight_decay=args.controller_l2_reg)
    for epoch in range(1, epochs+1):
        loss, mse, ce = controller_train(queue, model, optimizer)
        logging.info("epoch %04d train loss %.6f mse %.6f ce %.6f", epoch, loss, mse, ce)


def generate_synthetic_controller_data(model, exclude=[], maxn=1000):
    synthetic_input = []
    synthetic_target = []
    while len(synthetic_input) < maxn:
        synthetic_arch = utils.generate_arch(1, args.layers, args.num_ops)[0]
        if synthetic_arch not in exclude and synthetic_arch not in synthetic_input:
            synthetic_input.append(synthetic_arch)
    
    synthetic_dataset = utils.ControllerDataset(synthetic_input, None, False)      
    synthetic_queue = torch.utils.data.DataLoader(synthetic_dataset, batch_size=len(synthetic_dataset), shuffle=False, pin_memory=True)

    with torch.no_grad():
        model.eval()
        for sample in synthetic_queue:
            input = utils.move_to_cuda(sample['encoder_input'])
            _, _, _, predict_value = model.encoder(input)
            synthetic_target += predict_value.data.squeeze().tolist()
    assert len(synthetic_input) == len(synthetic_target)
    return synthetic_input, synthetic_target


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

    args.device_count = torch.cuda.device_count() if torch.cuda.is_available() else 1
    args.lr = args.lr
    args.batch_size = args.batch_size
    args.eval_batch_size = args.eval_batch_size
    args.width_stages = [int(val) for val in args.width_stages.split(',')]
    args.n_cell_stages = [int(val) for val in args.n_cell_stages.split(',')]
    args.stride_stages = [int(val) for val in args.stride_stages.split(',')]
    args.num_class = 1000
    args.num_ops = len(utils.OPERATIONS)
    args.controller_vocab_size = 1 + args.num_ops
    args.controller_source_length = args.layers
    args.controller_encoder_length = args.layers
    args.controller_decoder_length = args.controller_source_length

    logging.info("args = %s", args)
    
    if args.arch_pool is not None:
        logging.info('Architecture pool is provided, loading')
        with open(args.arch_pool) as f:
            archs = f.read().splitlines()
            archs = list(map(lambda x:list(map(int, x.strip().split())), archs))
            child_arch_pool = archs
    else:
        child_arch_pool = None

    train_queue, valid_queue, model, train_criterion, eval_criterion, optimizer, scheduler = build_imagenet(None, None, valid_num=5000, epoch=-1)

    controller = NAO(
        args.controller_encoder_layers,
        args.controller_mlp_layers,
        args.controller_decoder_layers,
        args.controller_vocab_size,
        args.controller_hidden_size,
        args.controller_mlp_hidden_size,
        args.controller_dropout,
        args.controller_encoder_length,
        args.controller_source_length,
        args.controller_decoder_length,
    )
    controller = controller.cuda()
    logging.info("Encoder-Predictor-Decoder param size = %d", utils.count_parameters(controller))

    
    if child_arch_pool is None:
        logging.info('Architecture pool is not provided, randomly generating now')
        child_arch_pool = utils.generate_arch(args.controller_seed_arch, args.layers, args.num_ops)

    arch_pool = []
    arch_pool_valid_acc = []
    for controller_iteration in range(args.controller_iterations+1):
        logging.info('Iteration %d', controller_iteration+1)

        child_arch_pool_prob = None
        num_updates = 0
        max_num_updates = args.max_num_updates
        epoch = 1
        while True:
            lr = scheduler.get_lr()[0]
            logging.info('epoch %d lr %e', epoch, lr)
            # sample an arch to train
            train_acc, train_obj, num_updates = child_train(train_queue, model, optimizer, num_updates, child_arch_pool, child_arch_pool_prob, train_criterion)
            epoch += 1
            scheduler.step()
            if num_updates >= max_num_updates:
                break
    
        logging.info("Evaluate seed archs")
        arch_pool += child_arch_pool
        arch_pool_valid_acc = child_valid(valid_queue, model, arch_pool, eval_criterion)

        arch_pool_valid_acc_sorted_indices = np.argsort(arch_pool_valid_acc)[::-1]
        arch_pool = list(map(lambda x:arch_pool[x], arch_pool_valid_acc_sorted_indices))
        arch_pool_valid_acc = list(map(lambda x:arch_pool_valid_acc[x], arch_pool_valid_acc_sorted_indices))
        with open(os.path.join(args.output_dir, 'arch_pool.{}'.format(controller_iteration)), 'w') as fa:
            with open(os.path.join(args.output_dir, 'arch_pool.perf.{}'.format(controller_iteration)), 'w') as fp:
                for arch, perf in zip(arch_pool, arch_pool_valid_acc):
                    arch = ' '.join(map(str, arch))
                    fa.write('{}\n'.format(arch))
                    fp.write('{}\n'.format(perf))
        if controller_iteration == args.controller_iterations:
            break
                            
        # Train Encoder-Predictor-Decoder
        logging.info('Train Encoder-Predictor-Decoder')
        inputs = arch_pool
        min_val = min(arch_pool_valid_acc)
        max_val = max(arch_pool_valid_acc)
        targets = list(map(lambda x: (x - min_val) / (max_val - min_val), arch_pool_valid_acc))

        # Pre-train
        logging.info('Pre-train EPD')
        train_controller(controller, inputs, targets, args.controller_pretrain_epochs)
        logging.info('Finish pre-training EPD')
        # Generate synthetic data
        logging.info('Generate synthetic data for EPD')
        synthetic_inputs, synthetic_targets = generate_synthetic_controller_data(controller, inputs, args.controller_random_arch)
        if args.controller_up_sample_ratio:
            all_inputs = inputs * args.controller_up_sample_ratio + synthetic_inputs
            all_targets = targets * args.controller_up_sample_ratio + synthetic_targets
        else:
            all_inputs = inputs + synthetic_inputs
            all_targets = targets + synthetic_targets
        # Train
        logging.info('Train EPD')
        train_controller(controller, all_inputs, all_targets, args.controller_epochs)
        logging.info('Finish training EPD')

        # Generate new archs
        new_archs = []
        max_step_size = 100
        predict_step_size = 0.0
        topk_indices = np.argsort(all_targets)[:100]
        topk_archs = list(map(lambda x:all_inputs[x], topk_indices))
        infer_dataset = utils.ControllerDataset(topk_archs, None, False)
        infer_queue = torch.utils.data.DataLoader(
            infer_dataset, batch_size=len(infer_dataset), shuffle=False, pin_memory=True)
        while len(new_archs) < args.controller_new_arch:
            predict_step_size += 0.1
            logging.info('Generate new architectures with step size %.2f', predict_step_size)
            new_arch = controller_infer(infer_queue, controller, predict_step_size, direction='+')
            for arch in new_arch:
                if arch not in inputs and arch not in new_archs:
                    new_archs.append(arch)
                if len(new_archs) >= args.controller_new_arch:
                    break
            logging.info('%d new archs generated now', len(new_archs))
            if predict_step_size > max_step_size:
                break

        child_arch_pool = new_archs
        logging.info("Generate %d new archs", len(child_arch_pool))

    logging.info('Finish Searching')
    
    top_archs = arch_pool[:5]
    top_archs_perf = arch_pool_valid_acc[:5]
    with open(os.path.join(args.output_dir, 'arch_pool.final'), 'w') as fa:
        with open(os.path.join(args.output_dir, 'arch_pool.perf.final'), 'w') as fp:
            for arch, perf in zip(top_archs, top_archs_perf):
                arch = ' '.join(map(str, arch))
                fa.write('{}\n'.format(arch))
                fp.write('{}\n'.format(perf))
  

if __name__ == '__main__':
    main()
