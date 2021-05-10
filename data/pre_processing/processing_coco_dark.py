import os
from glob import glob
import numpy as np
from pycocotools.coco import COCO
import json
import subprocess
from tqdm import tqdm

# COCO-dark
def split_train_val_coco_dark(coco_dir, exdark_label_path, save_folder):
    # Load ExDark Label
    with open(exdark_label_path, 'r') as f:
        exdark_label = f.readlines()

    label_dict = {}
    for label in exdark_label:
        label_dict[str(label.strip().split(',')[0])] = str(label.strip().split(',')[1])

    # Base Folder
    annotation_dir = os.path.join(save_folder, 'annotations')
    train_dir = os.path.join(save_folder, 'train2017')
    val_dir = os.path.join(save_folder, 'val2017')

    os.makedirs(annotation_dir, exist_ok=True)
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # Load COCO dataset - train
    select_target_coco(list(label_dict.keys()), coco_dir, save_folder, type='train')

    # Load COCO dataset - val
    select_target_coco(list(label_dict.keys()), coco_dir, save_folder, type='val')


def select_target_coco(target_list, coco_dir, save_folder, type='train'):
    # Path
    ann_path = os.path.join(coco_dir, 'annotations', 'instances_%s2017' %type)
    save_path = os.path.join(save_folder, 'annotations', 'instances_%s2017' %type)

    image_old_dir = os.path.join(coco_dir, '%s2017' %type)
    image_new_dir = os.path.join(save_folder, '%s2017' %type)

    # Load COCO ANN file
    coco = COCO(ann_path)
    target_list = [int(target) for target in target_list]

    # Category
    category = []
    cat = coco.loadCats(ids=target_list)
    for cls in cat:
        category.append(cls)

    # Target Image List
    img_id_list = []
    for target in target_list:
        imgIDs = coco.getImgIds(catIds=target)
        img_id_list += imgIDs

    img_id_list = list(set(img_id_list))
    img = coco.loadImgs(ids=img_id_list)

    # Move Image File
    for img_dict in tqdm(img):
        old_path = os.path.join(image_old_dir, img_dict['file_name'])
        new_path = image_new_dir

        script = 'cp %s %s' %(old_path, new_path)
        subprocess.call(script, shell=True)

    # Annotation
    annId = coco.getAnnIds(catIds=target_list)
    ann = coco.loadAnns(ids=annId)

    # Load pre-exist JSON file
    with open(ann_path, 'r') as f:
        file = json.load(f)

    # Update Annotation
    file['images'] = img
    file['annotations'] = ann
    file['categories'] = category

    # Save
    with open(save_path, 'w') as f:
        json.dump(file, f)

if __name__=='__main__':
    split_train_val_coco_dark(coco_dir='/data/sung/dataset/coco',
                              exdark_label_path='/data/sung/dataset/low-illumination-dataset/exdark/exdark_labels.txt',
                              save_folder='/data/sung/dataset/low-illumination-dataset/coco_dark')


