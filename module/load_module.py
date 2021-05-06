import torch
import torch.nn as nn
from .network import IL_RobustNet, backbone_extractor, backbone_split
from .layers.resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from copy import deepcopy
import torch

def load_model(option):
    # Select Backbone
    backbone = backbone_extractor(resnet18(), target_layer='avgpool')
    backbone_1, backbone_2 = backbone_split(backbone, 'layer1')

    # Select Detector Head
    detector = None
    model = IL_RobustNet(backbone_1, backbone_2, detector)
    return model


def load_optimizer(option, params):
    optimizer = option.result['optim']['optimizer']
    lr = option.result['optim']['lr']
    weight_decay = option.result['optim']['weight_decay']

    if optimizer == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    elif optimizer == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    elif optimizer == "sgd":
        return torch.optim.SGD(params, lr=lr, weight_decay=weight_decay, nesterov=option.result['optim']['nesterov'], momentum=option.result['optim']['momentum'])
    else:
        raise('Selec proper optimizer')


def load_scheduler(option, optimizer):
    if option.result['train']['scheduler'] == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=option.result['train']['total_epoch'])
    elif option.result['train']['scheduler'] == 'step':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[48, 63, 80], gamma=0.2)
    elif option.result['train']['scheduler'] is None:
        scheduler = None
    else:
        raise('select proper scheduler')

    return scheduler

def load_loss(option):
    train_type = option.result['train']['train_type']
    if train_type == 'naive':
        criterion = nn.CrossEntropyLoss()
    else:
        raise('select proper train_type')

    return criterion

