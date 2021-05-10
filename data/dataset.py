from torchvision.transforms import transforms
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from random import shuffle
import torch
from glob import glob
from PIL import Image
import cv2
from pycocotools.coco import COCO
from .augmentations import SSDAugmentation, BaseTransform

## COCO dataset
def get_label_map(label_file):
    label_map = {}
    labels = open(label_file, 'r')
    for line in labels:
        ids = line.split(',')
        label_map[int(ids[0])] = int(ids[1])
    return label_map


class COCOAnnotationTransform(object):
    """Transforms a COCO annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes
    """
    def __init__(self, label_path):
        self.label_map = get_label_map(label_path)

    def __call__(self, target, width, height):
        """
        Args:
            target (dict): COCO target json annotation as a python dict
            height (int): height
            width (int): width
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class idx]
        """
        scale = np.array([width, height, width, height])
        res = []
        for obj in target:
            if 'bbox' in obj:
                bbox = obj['bbox']
                bbox[2] += bbox[0]
                bbox[3] += bbox[1]
                label_idx = self.label_map[obj['category_id']] - 1
                final_box = list(np.array(bbox)/scale)
                final_box.append(label_idx)
                res += [final_box]  # [xmin, ymin, xmax, ymax, label_idx]
            else:
                print("no bbox problem!")

        return res  # [[xmin, ymin, xmax, ymax, label_idx], ... ]


class COCODetection(Dataset):
    """`MS Coco Detection <http://mscoco.org/dataset/#detections-challenge2016>`_ Dataset.
    Args:
        root (string): Root directory where images are downloaded to.
        set_name (string): Name of the specific set of COCO images.
        transform (callable, optional): A function/transform that augments the
                                        raw images`
        target_transform (callable, optional): A function/transform that takes
        in the target (bbox) and transforms it.
    """

    def __init__(self, data_dir, type='train', transform=None):
        self.root = os.path.join(data_dir, '%s2017' %type)
        self.coco = COCO(os.path.join(data_dir, 'annotations', 'instances_%s2017.json' %type))
        self.ids = list(self.coco.imgToAnns.keys())

        self.transform = transform
        self.target_transform = COCOAnnotationTransform(label_path=os.path.join(data_dir, 'exdark_labels.txt'))


    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target).
                   target is the object returned by ``coco.loadAnns``.
        """
        im, gt, h, w = self.pull_item(index)
        return im, gt


    def __len__(self):
        return len(self.ids)


    def pull_item(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target, height, width).
                   target is the object returned by ``coco.loadAnns``.
        """
        img_id = self.ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)

        target = self.coco.loadAnns(ann_ids)
        path = os.path.join(self.root, self.coco.loadImgs(img_id)[0]['file_name'])
        img = cv2.imread(os.path.join(self.root, path))

        height, width, _ = img.shape
        if self.target_transform is not None:
            target = self.target_transform(target, width, height)

        if self.transform is not None:
            target = np.array(target)
            img, boxes, labels = self.transform(img, target[:, :4], target[:, 4])

            # to rgb
            img = img[:, :, (2, 1, 0)]

            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
        return torch.from_numpy(img).permute(2, 0, 1), target, height, width


    def pull_image(self, index):
        '''Returns the original image object at index in PIL form
        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.
        Argument:
            index (int): index of img to show
        Return:
            cv2 img
        '''
        img_id = self.ids[index]
        path = self.coco.loadImgs(img_id)[0]['file_name']
        return cv2.imread(os.path.join(self.root, path), cv2.IMREAD_COLOR)


    def pull_anno(self, index):
        '''Returns the original annotation of image at index
        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.
        Argument:
            index (int): index of img to get annotation of
        Return:
            list:  [img_id, [(label, bbox coords),...]]
                eg: ('001718', [('dog', (96, 13, 438, 332))])
        '''
        img_id = self.ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        return self.coco.loadAnns(ann_ids)


def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).
    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations
    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on
                                 0 dim
    """
    targets = []
    imgs = []
    for sample in batch:
        imgs.append(sample[0])
        targets.append(torch.FloatTensor(sample[1]))
    return torch.stack(imgs, 0), targets


## Paired LOL dataset
class LOL_dataset(Dataset):
    def __init__(self, data_dir, transform, type):
        self.data_dir = os.path.join(data_dir, type)

        self.data_name = [path.split('/')[-1] for path in glob(os.path.join(self.data_dir, 'high', '*.png'))]
        self.transform = transform

    def __len__(self):
        return len(self.data_name)

    def __getitem__(self, index):
        high_img = self.load_image(path=os.path.join(self.data_dir, 'high', self.data_name[index]))
        low_img = self.load_image(path=os.path.join(self.data_dir, 'low', self.data_name[index]))

        return high_img, low_img

    def load_image(self, path):
        img = Image.open(path).convert('RGB')
        img = self.transform(img)
        return img


## Load Dataset appropriate for train type
def load_data(option, type='train'):
    train_type = option.result['train']['train_type']
    cfg = option.result['detector']

    # Robust Dataset
    if 'robust' in train_type:
        transform = transforms.Compose([transforms.Resize((300, 300)),
                                        transforms.ToTensor(),
                                        transforms.Normalize([cfg['mean'][2]/255, cfg['mean'][1]/255, cfg['mean'][0]/255], [1, 1, 1])])
        robust_dataset = LOL_dataset(option.result['data']['robust_dir'], type, transform)
    else:
        robust_dataset = None

    # Detection-COCO-Dataset
    if 'coco-d' in train_type:
        if type == 'train':
            transform = SSDAugmentation(cfg['min_dim'], cfg['mean'])
        else:
            transform = BaseTransform(cfg['min_dim'], cfg['mean'])
        coco_dataset = COCODetection(option.result['data']['coco-d_dir'], type, transform)
    else:
        coco_dataset = None

    # Detection-EX-DARK-Dataset
    if 'ex-d' in train_type:
        if type == 'train':
            transform = SSDAugmentation(cfg['min_dim'], cfg['mean'])
        else:
            transform = BaseTransform(cfg['min_dim'], cfg['mean'])
        ex_dataset = COCODetection(option.result['data']['ex-d_dir'], type, transform)
    else:
        ex_dataset = None

    return robust_dataset, coco_dataset, ex_dataset


if __name__=='__main__':
    dset = COCODetection(data_dir='/data/sung/dataset/coco')
    print(dset.__getitem__(1))

