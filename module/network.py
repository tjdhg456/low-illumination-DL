import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from tqdm import tqdm
from collections import OrderedDict

def backbone_extractor(model, target_layer='avgpool'):
    backbone = []

    for name, param in model.named_children():
        if name == target_layer:
            break
        else:
            backbone.append((name, param))

    backbone = nn.Sequential(OrderedDict(backbone))
    return backbone


def backbone_split(backbone, split='layer1'):
    # initialization
    former = []
    later = []
    token = False
    for name, children in backbone.named_children():
        # Split the model into two parts (former / later)
        if token == False:
            former.append((name, children))
        else:
            later.append((name, children))

        # Whether to split or not
        if name == split:
            token = True

    # Wrapping with model
    former = nn.Sequential(OrderedDict(former))
    later = nn.Sequential(OrderedDict(later))
    return former, later


class Identity_Layer(nn.Module):
    def __init__(self):
        super(Identity_Layer, self).__init__()

    def forward(self, x):
        return x


class IL_RobustNet(nn.Module):
    def __init__(self, backbone_1, backbone_2, detector):
        super(IL_RobustNet, self).__init__()
        self.backbone_1 = backbone_1
        self.backbone_2 = backbone_2
        self.detector = detector

    def feature_extract(self, input):
        feature = self.backbone_1(input)
        return feature

    def object_detect(self, feature):
        feature = self.backbone_2(feature)
        out = self.detector(feature)
        return out

    def forward(self, input):
        feature = self.backbone_2(self.backbone_1(input))
        out = self.detector(feature)
        return out

    def freeze_backbone_1(self):
        for name, param in self.backbone_1.named_paramters():
            param.requires_grad = False

    def freeze_backbone_2(self):
        for name, param in self.backbone_2.named_paramters():
            param.requires_grad = False

    def freeze_detector(self):
        for name, param in self.detector.named_parameters():
            param.requires_grad = False

    def freeze_all(self):
        for name, param in self.backbone_1.named_parameters():
            param.requires_grad = False

        for name, param in self.backbone_2.named_parameters():
            param.requires_grad = False

        for name, param in self.detector.named_parameters():
            param.requires_grad = False

    def unfreeze_backbone_1(self):
        for name, param in self.backbone_1.named_paramters():
            param.requires_grad = True

    def unfreeze_backbone_2(self):
        for name, param in self.backbone_2.named_paramters():
            param.requires_grad = True

    def unfreeze_detector(self):
        for name, param in self.detector.named_parameters():
            param.requires_grad = True

    def unfreeze_all(self):
        for name, param in self.backbone_1.named_parameters():
            param.requires_grad = True

        for name, param in self.backbone_2.named_parameters():
            param.requires_grad = True

        for name, param in self.detector.named_parameters():
            param.requires_grad = True