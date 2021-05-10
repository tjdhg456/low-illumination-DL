import json
import numpy as np
import pickle
import argparse
import os
import neptune

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18

from module.load_module import load_model, load_loss, load_optimizer, load_scheduler
from utility.utils import config, train_module
from utility.earlystop import EarlyStopping

from data.dataset import load_data, detection_collate

from copy import deepcopy
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from utility.distributed import apply_gradient_allreduce, reduce_tensor


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    # initialize the process group
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def main(rank, option, resume, save_folder):
    # Basic Options
    resume_path = os.path.join(save_folder, 'last_dict.pt')

    num_gpu = len(option.result['train']['gpu'].split(','))

    total_epoch = option.result['train']['total_epoch']
    multi_gpu = len(option.result['train']['gpu'].split(',')) > 1
    if multi_gpu:
        ddp = option.result['train']['ddp']
    else:
        ddp = False

    scheduler = option.result['train']['scheduler']
    batch_size, pin_memory = option.result['train']['batch_size'], option.result['train']['pin_memory']

    # Logger
    if (rank == 0) or (rank == 'cuda'):
        neptune.init('sunghoshin/imp', api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI5MTQ3MjY2Yy03YmM4LTRkOGYtOWYxYy0zOTk3MWI0ZDY3M2MifQ==')
        exp_name, exp_num = save_folder.split('/')[-2], save_folder.split('/')[-1]
        neptune.create_experiment(params={'exp_name':exp_name, 'exp_num':exp_num},
                                  tags=['inference:False'])

    # Load Model
    model = load_model(option)
    patch_criterion, detection_criterion = load_loss(option, rank)
    save_module = train_module(total_epoch, model, patch_criterion, detection_criterion, multi_gpu)

    if resume:
        save_module.import_module(resume_path)
        model.load_state_dict(save_module.save_dict['model'][0])

    # Multi-Processing GPUs
    if ddp:
        setup(rank, num_gpu)
        torch.manual_seed(0)
        torch.cuda.set_device(rank)

        model.to(rank)
        model = DDP(model, device_ids=[rank])

        model = apply_gradient_allreduce(model)
        patch_criterion.to(rank)
        detection_criterion.to(rank)

    else:
        if multi_gpu:
            model = nn.DataParallel(model).to(rank)
        else:
            model = model.to(rank)

    # Optimizer and Scheduler
    if resume:
        # Load Optimizer
        optimizer = load_optimizer(option, model.parameters())
        optimizer.load_state_dict(save_module.save_dict['optimizer'][0])

        # Load Scheduler
        if scheduler is not None:
            scheduler = load_scheduler(option, optimizer)
            scheduler.load_state_dict(save_module.save_dict['scheduler'][0])

    else:
        optimizer = load_optimizer(option, model.parameters())
        if scheduler is not None:
            scheduler = load_scheduler(option, optimizer)

    # Early Stopping
    early_stop = option.result['train']['early']

    if early_stop:
        early = EarlyStopping(patience=option.result['train']['patience'])
    else:
        early = None


    # Dataset and DataLoader
    tr_robust_dataset, tr_coco_dataset, tr_ex_dataset = load_data(option, type='train')
    val_robust_dataset, val_coco_dataset, val_ex_dataset = load_data(option, type='val')

    if ddp:
        # Robust Dataset Loader
        if tr_robust_dataset is not None:
            tr_robust_sampler = torch.utils.data.distributed.DistributedSampler(dataset=tr_robust_dataset,
                                                                         num_replicas=num_gpu, rank=rank)
            val_robust_sampler = torch.utils.data.distributed.DistributedSampler(dataset=val_robust_dataset,
                                                                         num_replicas=num_gpu, rank=rank)


            tr_robust_loader = torch.utils.data.DataLoader(dataset=tr_robust_dataset, batch_size=batch_size,
                                                      shuffle=False, num_workers=4*num_gpu, pin_memory=pin_memory,
                                                      sampler=tr_robust_sampler)
            val_robust_loader = torch.utils.data.DataLoader(dataset=val_robust_dataset, batch_size=batch_size,
                                                      shuffle=False, num_workers=4*num_gpu, pin_memory=pin_memory,
                                                      sampler=val_robust_sampler)
        else:
            tr_robust_loader = None
            val_robust_loader = None

        # Detection-COCO-Dark-Dataset Loader
        if tr_coco_dataset is not None:
            tr_coco_sampler = torch.utils.data.distributed.DistributedSampler(dataset=tr_coco_dataset,
                                                                                num_replicas=num_gpu, rank=rank)

            val_coco_sampler = torch.utils.data.distributed.DistributedSampler(dataset=val_coco_dataset,
                                                                                 num_replicas=num_gpu, rank=rank)

            tr_coco_loader = torch.utils.data.DataLoader(dataset=tr_coco_dataset, batch_size=batch_size,
                                                           shuffle=False, num_workers=4 * num_gpu,
                                                           pin_memory=pin_memory,
                                                           sampler=tr_coco_sampler, collate_fn=detection_collate)
            val_coco_loader = torch.utils.data.DataLoader(dataset=val_coco_dataset, batch_size=batch_size,
                                                            shuffle=False, num_workers=4 * num_gpu,
                                                            pin_memory=pin_memory,
                                                            sampler=val_coco_sampler, collate_fn=detection_collate)
        else:
            tr_coco_loader = None
            val_coco_loader = None


        # Detection-EX-Dark-Dataset Loader
        if tr_ex_dataset is not None:
            tr_ex_sampler = torch.utils.data.distributed.DistributedSampler(dataset=tr_ex_dataset,
                                                                                num_replicas=num_gpu, rank=rank)

            val_ex_sampler = torch.utils.data.distributed.DistributedSampler(dataset=val_ex_dataset,
                                                                                 num_replicas=num_gpu, rank=rank)

            tr_ex_loader = torch.utils.data.DataLoader(dataset=tr_ex_dataset, batch_size=batch_size,
                                                           shuffle=False, num_workers=4 * num_gpu,
                                                           pin_memory=pin_memory,
                                                           sampler=tr_ex_sampler, collate_fn=detection_collate)
            val_ex_loader = torch.utils.data.DataLoader(dataset=val_ex_dataset, batch_size=batch_size,
                                                            shuffle=False, num_workers=4 * num_gpu,
                                                            pin_memory=pin_memory,
                                                            sampler=val_ex_sampler, collate_fn=detection_collate)
        else:
            tr_ex_loader = None
            val_ex_loader = None


    else:
        # Robust Dataset Loader
        if tr_robust_dataset is not None:
            tr_robust_loader = DataLoader(tr_robust_dataset, batch_size=batch_size, shuffle=True, pin_memory=pin_memory, num_workers=4*num_gpu)
            val_robust_loader = DataLoader(val_robust_dataset, batch_size=batch_size, shuffle=False, pin_memory=pin_memory, num_workers=4*num_gpu)
        else:
            tr_robust_loader = None
            val_robust_loader = None

        # Detection-COCO-Dark-Dataset Loader
        if tr_coco_dataset is not None:
            tr_coco_loader = DataLoader(tr_coco_dataset, batch_size=batch_size, shuffle=True, pin_memory=pin_memory, num_workers=4 * num_gpu, collate_fn=detection_collate)
            val_coco_loader = DataLoader(val_coco_dataset, batch_size=batch_size, shuffle=False, pin_memory=pin_memory, num_workers=4 * num_gpu, collate_fn=detection_collate)
        else:
            tr_coco_loader = None
            val_coco_loader = None

        # Detection-EX-Dark-Dataset Loader
        if tr_ex_dataset is not None:
            tr_ex_loader = DataLoader(tr_ex_dataset, batch_size=batch_size, shuffle=True, pin_memory=pin_memory, num_workers=4 * num_gpu, collate_fn=detection_collate)
            val_ex_loader = DataLoader(val_ex_dataset, batch_size=batch_size, shuffle=False, pin_memory=pin_memory, num_workers=4 * num_gpu, collate_fn=detection_collate)
        else:
            tr_ex_loader = None
            val_ex_loader = None


    # Mixed Precision
    mixed_precision = option.result['train']['mixed_precision']
    if mixed_precision:
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None


    # Training
    from module.trainer import robust_trainer
    early, save_module, option = robust_trainer.run(option, model, tr_robust_loader, tr_coco_loader, tr_ex_loader, \
                                                    val_robust_loader, val_coco_loader, val_ex_loader, optimizer, \
                                                    patch_criterion, detection_criterion, scaler, scheduler, early, \
                                                    save_folder, save_module, multi_gpu, rank, neptune)

    if ddp:
        cleanup()

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--save_dir', type=str, default='/HDD1/sung/checkpoint/')
    parser.add_argument('--exp_name', type=str, default='imagenet_norm')
    parser.add_argument('--exp_num', type=int, default=1)
    args = parser.parse_args()

    # Configure
    save_folder = os.path.join(args.save_dir, args.exp_name, str(args.exp_num))
    os.makedirs(save_folder, exist_ok=True)
    option = config(save_folder)
    option.get_all_config()

    # Resume Configuration
    resume = option.result['train']['resume']
    resume_path = os.path.join(save_folder, 'last_dict.pt')
    config_path = os.path.join(save_folder, 'last_config.json')

    if resume:
        if (os.path.isfile(resume_path) == False) or (os.path.isfile(config_path) == False):
            resume = False
        else:
            gpu = option.result['train']['gpu']

            option = config(save_folder)
            option.import_config(config_path)

            option.result['train']['gpu'] = gpu

    # Data Directory
    option.result['data']['robust_dir'] = os.path.join(option.result['data']['data_dir'], 'LOL')
    option.result['data']['coco-d_dir'] = os.path.join(option.result['data']['data_dir'], 'coco_dark')
    option.result['data']['ex-d_dir'] = os.path.join(option.result['data']['data_dir'], 'coco_exdark')

    # GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = option.result['train']['gpu']
    num_gpu = len(option.result['train']['gpu'].split(','))
    multi_gpu = num_gpu > 1
    if multi_gpu:
        ddp = option.result['train']['ddp']
    else:
        ddp = False

    if ddp:
        mp.spawn(main, args=(option,resume,save_folder,), nprocs=num_gpu, join=True)
    else:
        main('cuda', option, resume, save_folder)

