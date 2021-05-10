import torch

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

print(model.backbone)

