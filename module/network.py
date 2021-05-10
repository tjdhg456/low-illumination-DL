import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.nn import init
from tqdm import tqdm
from collections import OrderedDict

# Utility
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


################ Robust Network ##################
class IL_RobustNet(nn.Module):
    def __init__(self, option, detector):
        super(IL_RobustNet, self).__init__()
        self.option = option
        self.detector = detector
        self.features = []

        if 'robust' in self.option.result['train']['train_type']:
            self.patch_sampler = PatchSampler(use_mlp=option.result['train']['use_mlp'])

    # Patch Sampler
    def create_path_mlp(self, channels):
        self.patch_sampler.create_mlp(channels)

    def sample_patch(self, feats, ids=None):
        out_feats, out_ids = self.patch_sampler(feats, patch_ids=ids)
        return out_feats, out_ids

    # Forward
    def feature_extract(self, input):
        feature = self.detector.backbone(input)
        return feature

    def forward(self, input):
        out = self.detector(input)
        return out


    # Hook
    def get_features(self, _, inputs, outputs):
        self.features.append(outputs)

    def clear_features(self):
        self.features = []

    def get_hook(self, target_layers):
        for name, param in self.detector.backbone.named_children():
            if name in target_layers:
                setattr(self, 'hook_%s' %name, param.register_forward_hook(self.get_features))

    def remove_hook(self, target_layers):
        for name in target_layers:
            getattr(self, 'hook_%s' %name).remove()



################ Patch Sampler ##################
class PatchSampler(nn.Module):
    def __init__(self, use_mlp=False, init_type='normal', init_gain=0.02, nc=256):
        # potential issues: currently, we use the same patch_ids for multiple images in the batch
        super(PatchSampler, self).__init__()
        self.l2norm = Normalize(2)
        self.use_mlp = use_mlp
        self.nc = nc  # hard-coded
        self.init_type = init_type
        self.init_gain = init_gain

    def create_mlp(self, channels):
        for mlp_id, channel in enumerate(channels):
            input_nc = channel
            mlp = nn.Sequential(*[nn.Linear(input_nc, self.nc), nn.ReLU(), nn.Linear(self.nc, self.nc)])
            setattr(self, 'mlp_%d' % mlp_id, mlp)

        init_weights(self, self.init_type, self.init_gain)

    def forward(self, feats, num_patches=64, patch_ids=None):
        return_ids = []
        return_feats = []

        for feat_id, feat in enumerate(feats):
            B, H, W = feat.shape[0], feat.shape[2], feat.shape[3]
            feat_reshape = feat.permute(0, 2, 3, 1).flatten(1, 2)
            if num_patches > 0:
                if patch_ids is not None:
                    patch_id = patch_ids[feat_id]
                else:
                    patch_id = torch.randperm(feat_reshape.shape[1], device=feats[0].device)
                    patch_id = patch_id[:int(min(num_patches, patch_id.shape[0]))]  # .to(patch_ids.device)
                x_sample = feat_reshape[:, patch_id, :].flatten(0, 1)  # reshape(-1, x.shape[1])
            else:
                x_sample = feat_reshape
                patch_id = []

            if self.use_mlp:
                mlp = getattr(self, 'mlp_%d' % feat_id)
                x_sample = mlp(x_sample)

            return_ids.append(patch_id)
            x_sample = self.l2norm(x_sample)

            if num_patches == 0:
                x_sample = x_sample.permute(0, 2, 1).reshape([B, x_sample.shape[-1], H, W])
            return_feats.append(x_sample)
        return return_feats, return_ids


class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm + 1e-7)
        return out


def init_weights(net, init_type='normal', init_gain=0.02, debug=False):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if debug:
                print(classname)
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    net.apply(init_func)  # apply the initialization function <init_func>