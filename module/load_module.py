import torch
import torch.nn as nn
from .network import IL_RobustNet, backbone_extractor, backbone_split
from .layers.resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from .loss import PatchNCELoss

from ssd_module.ssd import build_ssd
from ssd_module.ssd_utils.multibox_loss import MultiBoxLoss
from copy import deepcopy
import torch

def load_model(option, phase='train'):
    train_option = option.result['train']
    network_type = option.result['network']['network_type']

    # Select Detector Head
    if network_type == 'ssd':
        detector = build_ssd(option, phase, num_classes=option.result['data']['num_classes'], size=300)
        detector.initialize()
    else:
        raise('Select Proper Detector Type')

    # Construct Model
    model = IL_RobustNet(option, detector)

    if 'robust' in train_option['train_type'] and train_option['use_mlp']:
        target_list = train_option['target_layers']
        target_channel_dict = {'0': 64, '5': 128, '10': 256, '17': 512}
        target_channel_list = [target_channel_dict[target] for target in target_list]

        model.create_path_mlp(channels=target_channel_list)

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

def load_loss(option, rank):
    train_type = option.result['train']['train_type']
    num_classes = option.result['data']['num_classes']
    variance = option.result['detector']['variance']

    # Robust Dataset
    if 'robust' in train_type:
        patch_criterion = PatchNCELoss(option)
    else:
        patch_criterion = None

    # Detection-COCO-Dataset
    if ('coco-d' in train_type) or ('ex-d' in train_type):
        detection_criterion = MultiBoxLoss(num_classes, 0.5, True, 0, True, 3, 0.5, False, variance, rank)
    else:
        detection_criterion = None

    return patch_criterion, detection_criterion

