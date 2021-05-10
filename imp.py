from module.layers.resnet import resnet18, resnet34, resnet50, resnet101, resnet152
import torch

# model = resnet18()
#
# x = torch.zeros([1, 3, 256, 256])
#
#
# print(getattr(model, 'layer4'))


class option(object):
    def __init__(self):
        self.result = {'detector' : {
                        "feature_maps": [38, 19, 10, 5, 3, 1],
                        "min_dim": 300,
                        "steps": [8, 16, 32, 64, 100, 300],
                        "min_sizes": [21, 45, 99, 153, 207, 261],
                        "max_sizes": [45, 99, 153, 207, 261, 315],
                        "aspect_ratios": [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
                        "variance": [0.1, 0.2],
                        "clip": True,
                        "name": "COCO"}
                      }

option = option()
from ssd_module.ssd import build_ssd
model = build_ssd(option, phase='train')

model.get_hook(['23'])
x = torch.zeros([1, 3, 300, 300])

print(model(x))

