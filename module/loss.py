import torch.nn as nn
import torch
import torch.nn.functional as F
import torch
from torch import nn


class PatchNCELoss(nn.Module):
    def __init__(self, option):
        super(PatchNCELoss, self).__init__()
        self.option = option
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.mask_dtype = torch.bool

    def forward(self, feat_q, feat_k):
        batch_patch_size = feat_q.shape[0]
        dim = feat_q.shape[1]

        feat_k = feat_k.detach()

        # pos logit
        l_pos = torch.bmm(feat_q.view(batch_patch_size, 1, -1), feat_k.view(batch_patch_size, -1, 1))
        l_pos = l_pos.view(batch_patch_size, 1)

        # reshape features to batch size
        batch_size = int(batch_patch_size / self.option.result['train']['num_patches'])
        feat_q = feat_q.view(batch_size, -1, dim)
        feat_k = feat_k.view(batch_size, -1, dim)
        npatches = self.option.result['train']['num_patches']
        l_neg_curbatch = torch.bmm(feat_q, feat_k.transpose(2, 1))

        # diagonal entries are similarity between same features, and hence meaningless.
        # just fill the diagonal with very small number, which is exp(-10) and almost zero
        diagonal = torch.eye(npatches, device=feat_q.device, dtype=self.mask_dtype)[None, :, :]
        l_neg_curbatch.masked_fill_(diagonal, -10.0) # Neglect the positive parts
        l_neg = l_neg_curbatch.view(-1, npatches)

        out = torch.cat((l_pos, l_neg), dim=1) / self.option.result['train']['temperature']
        loss = self.cross_entropy_loss(out, torch.zeros(out.size(0), dtype=torch.long, device=feat_q.device))
        return loss
