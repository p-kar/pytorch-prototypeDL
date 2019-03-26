import os
import pdb
import time
import random
import shutil
import warnings
import argparse
import tensorboardX
from tensorboardX import SummaryWriter

import torch
import torch.optim
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader

from utils import *
from models import CAE
from args import get_args
from logger import TensorboardXLogger

use_cuda = torch.cuda.is_available()

def run_iter(opts, data, target, model, criterion, device):
    data, target = data.to(device), target.to(device)
    output, R, R1, R2 = model(data)
    
    class_error = opts.lambda_class * criterion(output, target)
    ae_error = opts.lambda_ae * R
    error_1 = opts.lambda_1 * R1
    error_2 = opts.lambda_2 * R2
    loss = class_error + ae_error + error_1 + error_2
    pred = output.argmax(dim=1, keepdim=True)
    correct = pred.eq(target.view_as(pred)).sum().item()
    acc = correct / data.shape[0]

    return acc, loss, class_error, ae_error, error_1, error_2

def evaluate(opts, model, loader, criterion, device):
    model.eval()

    time_start = time.time()
    val_loss = 0.0
    val_class_error = 0.0
    val_ae_error = 0.0
    val_error_1 = 0.0
    val_error_2 = 0.0
    val_acc = 0.0
    num_batches = 0.0

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(loader):

            acc, loss, class_error, ae_error, error_1, error_2 = run_iter(opts, data, target, model, criterion, device)
            val_loss += loss.data.cpu().item()
            val_acc += acc
            val_class_error += class_error.data.cpu().item()
            val_ae_error += ae_error.data.cpu().item()
            val_error_1 += error_1.data.cpu().item()
            val_error_2 += error_2.data.cpu().item()
            num_batches += 1

    avg_valid_loss = val_loss / num_batches
    avg_valid_acc = val_acc / num_batches
    avg_class_error = val_class_error / num_batches
    avg_ae_error = val_ae_error / num_batches
    avg_error_1 = val_error_1 / num_batches
    avg_error_2 = val_error_2 / num_batches
    time_taken = time.time() - time_start

    return avg_valid_loss, avg_valid_acc, avg_class_error, avg_ae_error, avg_error_1, avg_error_2, time_taken


def train(opts):
    
    device = torch.device("cuda" if use_cuda else "cpu")

    if opts.mode == 'train_mnist':
        train_loader, valid_loader = get_mnist_loaders(opts.data_dir, opts.bsize, opts.nworkers, opts.sigma, opts.alpha)
        model = CAE(1, 10, 28, opts.n_prototypes)
    elif opts.mode == 'train_cifar':
        train_loader, valid_loader = get_cifar_loaders(opts.data_dir, opts.bsize, opts.nworkers, opts.sigma, opts.alpha)
        model = CAE(3, 10, 32, opts.n_prototypes)
    else:
        raise NotImplementedError('Unknown train mode')

    if opts.optim == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=opts.lr, weight_decay=opts.wd)
    else:
        raise NotImplementedError("Unknown optim type")
    criterion = nn.CrossEntropyLoss()

    start_n_iter = 0
    # for choosing the best model
    best_val_acc = 0.0

    model_path = os.path.join(opts.save_path, 'model_latest.net')
    if os.path.exists(model_path):
        # restoring training from save_state
        print ('====> Resuming training from previous checkpoint')
        save_state = torch.load(model_path, map_location='cpu')
        model.load_state_dict(save_state['state_dict'])
        start_n_iter = save_state['n_iter']
        best_val_acc = save_state['best_val_acc']
        opts = save_state['opts']
        opts.start_epoch = save_state['epoch'] + 1

    model = model.to(device)

    # for logging
    logger = TensorboardXLogger(opts.start_epoch, opts.log_iter, opts.log_dir)
    logger.set(['acc', 'loss', 'loss_class', 'loss_ae', 'loss_r1', 'loss_r2'])
    logger.n_iter = start_n_iter

    for epoch in range(opts.start_epoch, opts.epochs):
        model.train()
        logger.step()

        for batch_idx, (data, target) in enumerate(train_loader):
            acc, loss, class_error, ae_error, error_1, error_2 = run_iter(opts, data, target, model, criterion, device)

            # optimizer step
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), opts.max_norm)
            optimizer.step()

            logger.update(acc, loss, class_error, ae_error, error_1, error_2)

        val_loss, val_acc, val_class_error, val_ae_error, val_error_1, val_error_2, time_taken = evaluate(opts, model, valid_loader, criterion, device)
        # log the validation losses
        logger.log_valid(time_taken, val_acc, val_loss, val_class_error, val_ae_error, val_error_1, val_error_2)
        print ('')

        # Save the model to disk
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            save_state = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'n_iter': logger.n_iter,
                'opts': opts,
                'val_acc': val_acc,
                'best_val_acc': best_val_acc
            }
            model_path = os.path.join(opts.save_path, 'model_best.net')
            torch.save(save_state, model_path)
            prototypes = model.save_prototypes(opts.save_path, 'prototypes_best.png')
            x = torchvision.utils.make_grid(prototypes, nrow=10)
            logger.writer.add_image('Prototypes (best)', x, epoch)

        save_state = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'n_iter': logger.n_iter,
            'opts': opts,
            'val_acc': val_acc,
            'best_val_acc': best_val_acc
        }
        model_path = os.path.join(opts.save_path, 'model_latest.net')
        torch.save(save_state, model_path)
        prototypes = model.save_prototypes(opts.save_path, 'prototypes_latest.png')
        x = torchvision.utils.make_grid(prototypes, nrow=10)
        logger.writer.add_image('Prototypes (latest)', x, epoch)


opts = get_args()
train(opts)
