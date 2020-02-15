from __future__ import print_function
from __future__ import division

import os
import sys
import time
import datetime
import os.path as osp
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler
import torchvision.datasets as datasets
from torchvision import transforms
from torch.utils.data import DataLoader

from args import optimizer_kwargs
from torchreid import models
from torchreid.utils.avgmeter import AverageMeter
from torchreid.utils.loggers import Logger, RankLogger
from torchreid.utils.torchtools import count_num_param, open_all_layers, open_specified_layers,save_checkpoint
from torchreid.metrics.rank import evaluate_rank
from torchreid.optim.optimizer import build_optimizer
from torchreid.regularizers import get_regularizer
import torch.multiprocessing as mp
import argparse
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.manifold import TSNE
import random

import logging
logging.basicConfig(level=os.environ.get('LOGLEVEL', 'CRITICAL'))

# global variables
best_acc1 = 0

os.environ['TORCH_HOME'] = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '.torch'))


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--gpu-devices', default='0, 1, 2, 3', type=str, help='gpu device ids for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--gpu', type=int, default=None)
    parser.add_argument('--root', type=str, default='/media/ddj2/ce611f70-968b-4316-9547-9bc9cf931d32/V20200108/zhejiang_train')
    parser.add_argument('--save-dir', type=str, default='/media/ddj2/ce611f70-968b-4316-9547-9bc9cf931d32/V20200108/torch_save_abd')
    parser.add_argument('--multiprocessing-distributed', type=bool, default=True)
    parser.add_argument('--distributed', type=bool, default=False)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--num-classes', type=int, default=1001)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--resume', type=str, default='', metavar='PATH')
    parser.add_argument('--start-epoch', type=int, default=0)
    parser.add_argument('--step-size', type=int, default=(20, 40))
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--fixbase-epoch', type=int, default=10)
    parser.add_argument('--start-eval', type=int, default=0)
    parser.add_argument('--eval-freq', type=int, default=-1)
    parser.add_argument('--dist-url', default='tcp://127.0.0.1:23456', type=str)
    parser.add_argument('--dist-backend', default='nccl', type=str)
    parser.add_argument('--open-layers', type=str, default=['classifier'])
    parser.add_argument('--max-epoch', type=int, default=80)
    parser.add_argument('--rank', type=int, default=-1)
    parser.add_argument('--world-size', type=int, default=1)
    parser.add_argument('--print-freq', type=int, default=10)
    parser.add_argument('--height', type=int, default=672)
    parser.add_argument('--width', type=int, default=672)

    # # Optimizer Settings
    parser.add_argument('--optim', type=str, default='adam')
    parser.add_argument('--criterion', type=str, default='htri')
    parser.add_argument('--label-smooth', type=bool, default=True)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    # sgd
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--sgd-dampening', default=0, type=float)
    parser.add_argument('--sgd-nesterov', action='store_true', help="whether to enable sgd's Nesterov momentum")
    # rmsprop
    parser.add_argument('--rmsprop-alpha', default=0.99, type=float, help="rmsprop's smoothing constant")
    # adam/amsgrad
    parser.add_argument('--adam-beta1', default=0.9, type=float, help="exponential decay rate for adam's first moment")
    parser.add_argument('--adam-beta2', default=0.999, type=float, help="exponential decay rate for adam's second moment")

    # Hard Triplet Loss
    parser.add_argument('--margin', type=float, default=1.2, help="margin for triplet loss")
    parser.add_argument('--num-instances', type=int, default=4, help="number of instances per identity")
    parser.add_argument('--lambda-xent', type=float, default=1, help="weight to balance cross entropy loss")
    parser.add_argument('--lambda-htri', type=float, default=0.1, help="weight to balance hard triplet loss")

    # Branches
    parser.add_argument('--compatibility', action='store_true')
    parser.add_argument('--branches', nargs='+', type=str, default=['global', 'abd'])
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--global-dim', type=int, default=1024)
    parser.add_argument('--global-max-pooling', action='store_true')
    parser.add_argument('--abd-dim', type=int, default=1024)
    parser.add_argument('--abd-np', type=int, default=2)
    parser.add_argument('--abd-dan', nargs='+', type=str, default=['cam', 'pam'])
    parser.add_argument('--abd-dan-no-head', action='store_true')
    parser.add_argument('--shallow-cam', type=bool, default=True)
    parser.add_argument('--np-dim', type=int, default=1024)
    parser.add_argument('--np-np', type=int, default=2)
    parser.add_argument('--np-with-global', action='store_true')
    parser.add_argument('--np-max-pooling', action='store_true')
    parser.add_argument('--dan-dim', type=int, default=1024)
    parser.add_argument('--dan-dan', nargs='+', type=str, default=[])
    parser.add_argument('--dan-dan-no-head', action='store_true')
    parser.add_argument('--use-of', type=bool, default=True)
    parser.add_argument('--of-beta', type=float, default=1e-6)
    parser.add_argument('--of-start-epoch', type=int, default=23)
    parser.add_argument('--of-position', nargs='+', type=str, default=['before', 'after', 'cam', 'pam', 'intermediate'])
    parser.add_argument('--use-ow', type=bool, default=True)
    parser.add_argument('--ow-beta', type=float, default=1e-3)
    parser.add_argument('--arch', type=str, default='resnet50')
    parser.add_argument('--flip-eval', type=bool, default=True)

    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    # cudnn.deterministic = True

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        world_size = 1
        world_size = ngpus_per_node * world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        main_worker(args.gpu, ngpus_per_node, args)


def get_criterion(num_classes: int, use_gpu: bool, args):
    if args.criterion == 'htri':

        from torchreid.losses.hard_mine_triplet_loss import ABD_TripletLoss
        criterion = ABD_TripletLoss(num_classes, vars(args), use_gpu)

    elif args.criterion == 'xent':

        from torchreid.losses.cross_entropy_loss import CrossEntropyLoss
        criterion = CrossEntropyLoss(num_classes, use_gpu=use_gpu, label_smooth=args.label_smooth)
    else:
        raise RuntimeError('Unknown criterion {}'.format(args.criterion))

    return criterion


def main_worker(gpu, ngpus_per_node, args):
    # Data-loader
    # TODO:args
    global best_acc1
    args.gpu = gpu
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    train_dir = osp.join(args.root, 'train')
    val_dir = osp.join(args.root, 'val')

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    train_dataset = datasets.ImageFolder(
        train_dir,
        transform
    )
    val_dataset = datasets.ImageFolder(
        val_dir,
        transform
    )
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None
    trainloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.num_workers,
        sampler=train_sampler
    )
    valloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    # Initialize model
    # models.resnet_orig is the origin single-output resnet50
    # models.resnet is the multibranch resnet50 ==>ABD-Net
    # here is the ABD-Net
    model = models.build_model(name='abd_resnet50',
                              num_classes=args.num_classes,
                              loss={'xent'},
                              use_gpu=True,
                              args=vars(args))
    print("Model size: {:.3f} M".format(count_num_param(model)))

    if args.distributed:
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    # Criterion, Regularizer, Optimizer, Scheduler
    criterion = get_criterion(num_classes=args.num_classes, use_gpu=True, args=args)
    regularizer = get_regularizer(vars(args))
    optimizer = build_optimizer(model, **optimizer_kwargs(args))
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=args.step_size)

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    cudnn.benchmark = True
    start_time = time.time()
    train_time = 0
    print("==> Start training")

    if args.fixbase_epoch > 0:
        oldenv = os.environ.get('sa', '')
        os.environ['sa'] = ''
        print("Train {} for {} epochs while keeping other layers frozen".format(args.open_layers, args.fixbase_epoch))
        initial_optim_state = optimizer.state_dict()

        for epoch in range(args.fixbase_epoch):
            if args.distributed:
                train_sampler.set_epoch(epoch)
            start_train_time = time.time()
            train(epoch, model, criterion, regularizer, optimizer, trainloader, True, args, fixbase=True)
            train_time += round(time.time() - start_train_time)

        print("Done. All layers are open to train for {} epochs".format(args.max_epoch))
        optimizer.load_state_dict(initial_optim_state)
        os.environ['sa'] = oldenv

    for epoch in range(args.start_epoch, args.max_epoch):
        start_train_time = time.time()

        train(epoch, model, criterion, regularizer, optimizer, trainloader, True, args, fixbase=False)
        train_time += round(time.time() - start_train_time)

        state_dict = model.module.state_dict()

        save_checkpoint({
            'state_dict': state_dict,
            'rank1': 0,
            'epoch': epoch,
        }, osp.join(args.save_dir, 'checkpoint_ep' + str(epoch + 1) + '.pth.tar'), False)

        scheduler.step()

        if (epoch + 1) > args.start_eval and args.eval_freq > 0 and (epoch + 1) % args.eval_freq == 0 or (
                epoch + 1) == args.max_epoch:
            print("==> Test")
            print("Evaluating ...")
            acc1 = validate(valloader, model, criterion)
            is_best = acc1 > best_acc1
            best_acc1 = max(acc1, best_acc1)
            state_dict = model.module.state_dict()
            if args.rank % ngpus_per_node == 0:
                if best_acc1 < acc1:
                    print('Save!', best_acc1, acc1)
                    save_checkpoint({
                        'state_dict': state_dict,
                        'best_acc1': best_acc1,
                        'epoch': epoch,
                    }, osp.join(args.save_dir, 'checkpoint_best.pth.tar'), False)
                    best_acc1 = acc1

    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    train_time = str(datetime.timedelta(seconds=train_time))
    print("Finished. Total elapsed time (h:m:s): {}. Training time (h:m:s): {}.".format(elapsed, train_time))


def train(epoch, model, criterion, regularizer, optimizer, trainloader, use_gpu, args, fixbase=False):
    if not fixbase and args.use_of and epoch >= args.of_start_epoch:
        print('Using OF Penalty')
    from torchreid.losses.of_penalty import OFPenalty
    # args
    of_penalty = OFPenalty(vars(args))

    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    model.train()

    if fixbase or args.fixbase:
        open_specified_layers(model, args.open_layers)
    else:
        open_all_layers(model)

    end = time.time()
    for batch_idx, (imgs, target) in enumerate(trainloader):
        try:
            limited = float(os.environ.get('limited', None))
        except (ValueError, TypeError):
            limited = 1
        if not fixbase and (batch_idx + 1) > limited * len(trainloader):
            break
        data_time.update(time.time() - end)
        if args.gpu is not None:
            imgs = imgs.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)
        outputs = model(imgs)
        loss = criterion(outputs, target)
        if not fixbase:
            reg = regularizer(model)
            loss += reg
        if not fixbase and args.use_of and epoch >= args.of_start_epoch:
            penalty = of_penalty(outputs)
            loss += penalty

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        batch_time.update(time.time() - end)

        losses.update(loss.item(), target.size(0))

        if (batch_idx + 1) % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.4f} ({data_time.avg:.4f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                epoch + 1, batch_idx + 1, len(trainloader), batch_time=batch_time,
                data_time=data_time, loss=losses))

        end = time.time()


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def validate(val_loader, model, criterion):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    batch_time = AverageMeter()
    # switch to evaluate mode
    model.eval()
    os.environ['fake'] = '1'

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda()
            target = target.cuda()

            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            # for ABD-Net, the model has many outputs
            # now let the xent part be the input of accuracy func
            acc1, acc5 = accuracy(output[1][0], target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg


if __name__ == '__main__':
    main()
