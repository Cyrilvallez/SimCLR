#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 10:39:37 2022

@author: cyrilvallez
"""

import torch
from torch.nn import SyncBatchNorm
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
from datetime import datetime
import argparse
import os

from finetuning.simclr import SimCLR
from finetuning.nt_xent import NT_Xent
from finetuning.lars import LARS
from finetuning.transforms import ImageDataset


def get_optimizer(model, epochs, optimizer, learning_rate, weight_decay,
                  momentum, nesterov):
    """
    Returns the optimizer and scheduler from the parameters.

    Parameters
    ----------
    model : torch.nn.Module
        The model.
    epochs : int
        The number of epochs to train for.
    optimizer : str
        The optimizer. Either `lars` or `adam`.
    learning_rate : float
        The learning rate. 
    weight_decay : float
        Weight decay to apply to every parameter.
    momentum : float
        Momentum to use in case the optimizer is `lars`.
    nesterov : bool
        If `True`, nesterov mentum will be used if optimizer is `lars`. 

    Returns
    -------
    optimizer : torch.optim.Optimizer
        The optimizer.
    scheduler : torch.optim.lr_scheduler
        Cosine decay scheduler.

    """
    
    # This is correct to give all params to the optimizer even if
    # some will be frozen during fine-tuning
    params = model.parameters()
        
    if optimizer == 'lars':
        optimizer = LARS(params, lr=learning_rate, momentum=momentum,
                         weight_decay=weight_decay, nesterov=nesterov)
        
    elif optimizer == 'adam':
        optimizer = torch.optim.Adam(params, lr=learning_rate,
                                     weight_decay=weight_decay)
    
    scheduler = CosineAnnealingLR(optimizer, epochs)
    
    return optimizer, scheduler



def train_one_epoch(model, train_dataloader, criterion, optimizer,
                    local_rank, distributed):
    """
    Train the model for one epoch.

    Parameters
    ----------
    model : torch.nn.Module
        The model.
    train_dataloader : torch.utils.data.Dataloader
        Dataloader containing the train set.
    criterion : torch.nn.Module
        The loss function.
    optimizer : torch.optim.Optimizer
        The optimizer used.
    local_rank : int
        The local rank of the process inside the current node.
    distributed : bool
        Whether we are performing sitributed training or not.

    Returns
    -------
    float
        The loss of the training set (averaged and reduced over all processes).

    """
        
    running_average_loss = 0.
    
    for step, batch in enumerate(train_dataloader):
        
        # Get batch and set devices
        x1, x2 = batch
        x1 = x1.cuda(local_rank)
        x2 = x2.cuda(local_rank)
        
        # clear the gradients
        optimizer.zero_grad()
        
        # Forward pass
        h1, h2 = model(x1, x2)
        
        # Compute the loss and backward pass
        loss = criterion(h1, h2)
        loss.backward()
        
        # Adjust the weights
        optimizer.step()
        
        # Update the running average
        running_average_loss += loss.item()
        
    # Get the average of the loss and converts to tensor
    running_average_loss /= (step+1)
    running_average_loss = torch.tensor(running_average_loss,
                                        dtype=loss.dtype,
                                        device=loss.device)
        
    # Get average of running losses if distributed training
    if distributed:
        dist.all_reduce(running_average_loss,
                        op=dist.ReduceOp.AVG,
                        async_op=False)
        
    return running_average_loss.item()



def validate_one_epoch(model, val_dataloader, criterion, local_rank):
    """
    Perform validation for one epoch.

    Parameters
    ----------
    model : torch.nn.Module
        The model.
    val_dataloader : torch.utils.data.Dataloader
        Dataloader corrresponding to the validation set.
    criterion : torch.nn.Module
        The loss function.
    local_rank : int
        The local rank of the process inside the current node.

    Returns
    -------
    val_loss : float
        The loss on the validation set.

    """
    
    val_loss = 0.
    
    with torch.no_grad():
        
        for step, batch in enumerate(val_dataloader):
            
            # Get batch and set devices
            x1, x2 = batch
            x1 = x1.cuda(local_rank)
            x2 = x2.cuda(local_rank)
            
            # Forward pass
            h1, h2 = model(x1, x2)
            
            # Compute the loss and backward pass
            loss = criterion(h1, h2)
            val_loss += loss.item()
            
        # Get the average of the loss 
        val_loss /= (step+1)
            
        return val_loss
    
        

def train(model, epochs, train_dataloader, val_dataloader, criterion, optimizer,
          scheduler, writer, sampler, local_rank):
    """
    Train the model for the given number of epochs, and log the results.

    Parameters
    ----------
    model : torch.nn.Module
        The model.
    epochs : int
        The number of epochs to train for.
    train_dataloader : torch.utils.data.Dataloader
        Dataloader containing the train set.
    val_dataloader : torch.utils.data.Dataloader
        Dataloader corrresponding to the validation set. If `None`, no validation
        is performed.
    criterion : torch.nn.Module
        The loss function.
    optimizer : torch.optim.Optimizer
        The optimizer used.
    scheduler : torch.optim.lr_scheduler
        Scheduler used. If `None`, no scheduling is applied.
    writer : torch.utils.tensorboard.SummaryWriter
        The writer where to log the results.
    sampler : torch.utils.data.distributed.DistributedSampler
        The sampler in case of distributed training.
    local_rank : int
        The local rank of the process inside the current node.

    Returns
    -------
    None.

    """
    
    
    if dist.is_available() and dist.is_initialized():
        distributed = True
        rank = dist.get_rank()
    else:
        distributed = False
        rank = 0
        
    # We need to compute the gradients
    torch.set_grad_enabled(True)
    
    for epoch in range(epochs):
        
        # Set the epoch of the sampler (needed for correct shuffling)
        if distributed:
            sampler.set_epoch(epoch)
        
        # Train
        model.train(True)
        train_loss = train_one_epoch(model, train_dataloader, criterion, optimizer,
                                     local_rank, distributed)
        
        if scheduler is not None:
            # Get actual learning rate
            lr = scheduler.get_last_lr()[0]
        
            # Update learning rate
            scheduler.step()
        else:
            lr = optimizer.param_groups[0]['lr']
        
        # Eventually validate (but only on one GPU in case of distributed training)
        if val_dataloader is not None:
            if rank == 0:
                model.eval()
                val_loss = validate_one_epoch(model, val_dataloader, criterion,
                                              local_rank)
                
        if rank == 0:
            # Print a summary of current epoch
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            if val_dataloader is not None:
                print(f'{timestamp} : Epoch {epoch} --- train_loss : {train_loss:.3f}, val_loss : {val_loss:.3f}')
            else:
                print(f'{timestamp} : Epoch {epoch} --- train_loss : {train_loss:.3f}')
                   
            # Log quantities
            writer.add_scalar('Loss/train', train_loss, epoch)
            if val_dataloader is not None:
                writer.add_scalar('Loss/validation', val_loss, epoch)
            writer.add_scalar("Misc/learning_rate", lr, epoch)
                
            # Save the actual model
            outer_folder, inner_folder = writer.log_dir.rsplit('/', 1)
            path = outer_folder + '_models/' + inner_folder
            path += f'/epoch_{epoch+1}.pth'
            if distributed:
                model.module.save(path)
            else:
                model.save(path)
        
        # Synchonize processes as process on rank 0 should have an overhead
        # from validation, saving etc...
        if distributed:
            dist.barrier()
            
            
            
def setup(rank, args):
    """
    Initializes the process group.

    Parameters
    ----------
    rank : int
        The global rank of the process.
    args : Namespace
        Namespace containing all parameters for a run. See function `parse_args` 
        for details on every member.

    Returns
    -------
    None.

    """
    
    os.environ['MASTER_ADDR'] = args.master_address
    os.environ['MASTER_PORT'] = args.master_port

    # initialize the process group
    dist.init_process_group('nccl', rank=rank, world_size=args.world_size)    

        
            
def main(local_rank, args):
    """
    Main function running all the training from the parameters contained in
    `args`.

    Parameters
    ----------
    local_rank : int
        The local rank of the process inside the current node.
    args : Namespace
        Namespace containing all parameters for a run. See function `parse_args` 
        for details on every member.

    Returns
    -------
    None.

    """
    
    # global rank
    rank = args.node_index * args.gpus + local_rank
    
    # initialize the process group
    if args.world_size > 1:
        setup(rank, args)
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Model
    path = None if args.model == 'original' else args.model
    model = SimCLR.load(path, args.arch_depth, args.arch_width, args.arch_sk)
    
    # Configure the model
    model = model.cuda(local_rank)
    if args.world_size > 1:
        # converts to synchronized batchnorm layers
        model = SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[local_rank])
                   
    # Optimizer and scheduler
    optimizer, scheduler = get_optimizer(model, args.epochs, args.optimizer, 
                                         args.lr, args.weight_decay, args.momentum,
                                         args.nesterov)
    if args.scheduler == 'False':
        scheduler = None
        
    # Loss
    criterion = NT_Xent(args.temperature)
    
    # Datasets
    train_dataset = ImageDataset(args.train_dataset, args.size, args.jitter)
    if args.val_dataset != 'None':
        val_dataset = ImageDataset(args.val_dataset, args.size, args.jitter)
    
    # Create the dataloaders
    if args.world_size > 1:
        train_sampler = DistributedSampler(train_dataset, num_replicas=args.world_size,
                                           rank=rank, shuffle=True, drop_last=False)
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,
                                      sampler=train_sampler, num_workers=args.workers,
                                      pin_memory=True, drop_last=True)
    else:
        train_sampler = None
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,
                                      shuffle=True, num_workers=args.workers,
                                      pin_memory=True, drop_last=True)
    if args.val_dataset != 'None':
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size,
                                    shuffle=False, num_workers=args.workers,
                                    pin_memory=True, drop_last=False)
    else:
        val_dataloader = None
        
    # Configure the writer (will be only used by rank 0)
    if rank == 0:
        timestamp = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
        folder = args.log_dir
        log_dir = folder + '/' + timestamp
        writer = SummaryWriter(log_dir)
        # Create folder where we will save the models
        os.makedirs(folder + '_models/' + timestamp, exist_ok=True)
    else:
        writer = None
        
    if rank == 0:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f'{timestamp} : Starting training for {args.epochs} epochs.')
        
    # Perform training
    train(model, args.epochs, train_dataloader, val_dataloader, criterion, optimizer,
              scheduler, writer, train_sampler, local_rank)
    
    if rank == 0:
        print('Training ended.')
        
    # Destroy process group
    if args.world_size > 1:
        dist.destroy_process_group()
    


def parse_args():
    """
    Parse all arguments from the command line.

    Returns
    -------
    args : Namespace
        Namespace containing all the arguments.

    """
    
    parser = argparse.ArgumentParser(description='SimCLR finetuning')
    
    # Transform and dataset arguments
    parser.add_argument('--train_dataset', type=str, required=True,
                        help='Path to the dataset for training.')
    parser.add_argument('--val_dataset', type=str, default='None',
                        help='Path to the dataset for validation.')
    parser.add_argument('--size', type=int, default=224,
                        help='Size for resizing the images to.')
    parser.add_argument('--jitter', type=float, default=1.,
                        help='Jitter strength for color data augmentation.')
    
    # Training arguments
    parser.add_argument('--optimizer', type=str, default='lars', choices=['lars', 'adam'],
                        help='The optimizer used.')
    parser.add_argument('--lr', type=float, default=0., 
                        help='The base learning rate. Give 0 to use a square root rule based on batch size')
    parser.add_argument('--epochs', type=int, default=100, 
                        help='The number of epochs to perform.')
    parser.add_argument('--batch_size', type=int, default=32, 
                        help=('The batch size per GPU. Multiply by the number of'
                              ' GPUs and nodes you provide to get the total batch size'))
    parser.add_argument('--temperature', type=float, default=0.1, 
                        help='The temperature for the loss.')
    parser.add_argument('--weight_decay', type=float, default=1e-6, 
                        help='Weight decay.')
    parser.add_argument('--momentum', type=float, default=0.9, 
                        help='Momentum if the optimizer is `lars`.')
    parser.add_argument('--nesterov', type=str, default='False', choices=['False', 'True'],
                        help='Whether to apply nesterov momentum if optimizer is `lars`.')
    parser.add_argument('--scheduler', type=str, default='True', choices=['False', 'True'],
                        help='Whether to use a cosine scheduler.')
    
    # Config arguments
    parser.add_argument('--nodes', type=int, default=1, 
                        help='The number of nodes on which to run.')
    parser.add_argument('--node_index', type=int, default=0,
                        help='The index of the current node (between 0 and `nodes - 1`')
    parser.add_argument('--gpus', type=int, default=8,
                        help='The number of GPUs to use per node.')
    parser.add_argument('--workers', type=int, default=8,
                        help='The number of workers per GPUs to use.')
    parser.add_argument('--log_dir', type=str, required=True,
                        help='Where to save the results.')
    parser.add_argument('--master_address', type=str, default='localhost',
                        help='Master adress of the main node.')
    parser.add_argument('--master_port', type=str, default='12355',
                        help='Master port of the main node.')
    parser.add_argument('--seed', type=int, default=123,
                        help='Random seed for the number generators.')
    
    
    args = parser.parse_args()
    
    args.world_size = args.nodes*args.gpus
    
    # Remove last `/` if present in log_dir
    if args.log_dir[-1] == '/':
        args.log_dir = args.log_dir[0:-1]
    # if lr is 0, we use a square root rule
    if args.lr == 0:
        args.lr = 0.005*np.sqrt(args.batch_size*args.world_size)
    
    return args

