import datetime
from PIL import Image
import os
from glob import glob
import json
import subprocess
from tqdm import tqdm

INFO = {
"description": "Low-Illumination-DataSet",
"url": "",
"version": "0.1.0",
"year": 2021,
"contributor": "sungho shin",
"date_created": datetime.datetime.utcnow().isoformat(' ')
}

LICENSES = [
{
    "id": 1,
    "name": "Attribution-NonCommercial-ShareAlike License",
    "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
}
]

# Ex-Dark
def split_train_val_exdark(exdark_dir, exdark_meta_path, save_folder):
    # Load ExDark Meta Path
    with open(exdark_meta_path, 'r') as f:
        exdark_meta = f.readlines()
        exdark_meta = exdark_meta[1:]

    # Load the MetaDict for matching file_name and train_type
    meta_dict = {}
    for meta in exdark_meta:
        file_name = meta.strip().split(' ')[0]
        type = meta.strip().split(' ')[4]
        if file_name not in meta_dict.keys():
            meta_dict[file_name] = int(type)
        else:
            print(file_name)

    # Base Folder
    annotation_dir = os.path.join(save_folder, 'annotations')
    train_dir = os.path.join(save_folder, 'train2017')
    val_dir = os.path.join(save_folder, 'val2017')
    test_dir = os.path.join(save_folder, 'test2017')

    os.makedirs(annotation_dir, exist_ok=True)
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Move the File
    img_list = glob(os.path.join(exdark_dir, 'image', '*', '*'))
    for img in tqdm(img_list):
        file_name = img.split('/')[-1]
        if meta_dict[file_name] == 1:
            type = 'train'
        elif meta_dict[file_name] == 2:
            type = 'val'
        elif meta_dict[file_name] == 3:
            type = 'test'
        else:
            raise('Not proper train type')

        old_path = img
        new_dir = os.path.join(save_folder, '%s2017' %type)
        script = 'cp %s %s' %(old_path, new_dir)
        subprocess.call(script, shell=True)

## Convert the ExDark dataset into COCO format
def create_image_info(ann_path, img_name, image_id):
    # Load Image and Get Size Information
    file_name = os.path.join(os.path.dirname(ann_path.replace('annotation', 'image')), img_name)
    W, H = Image.open(file_name).size

    image = {
        "id": image_id,
        "width": W,
        "height": H,
        "file_name": file_name.split('/')[-1],
        "license": 1
    }
    return image

def exdark_coco_converter(exdark_dir, exdark_label_path, exdark_meta_path, save_folder):
    # Load ExDark Meta Path
    with open(exdark_meta_path, 'r') as f:
        exdark_meta = f.readlines()
        exdark_meta = exdark_meta[1:]

    # Load the MetaDict for matching file_name and train_type
    meta_dict = {}
    file_exe_dict = {}
    for meta in exdark_meta:
        file_name = meta.strip().split(' ')[0]
        type = meta.strip().split(' ')[4]
        if file_name not in meta_dict.keys():
            meta_dict[file_name] = int(type)
            file_exe_dict[file_name.split('.')[0]] = file_name.split('.')[-1]
        else:
            print(file_name)

    # Load EX-DARK label
    with open(exdark_label_path, 'r') as f:
        exdark_label = f.readlines()

    # Load Label Files
    categories = []
    label_dict = {}
    for label in exdark_label:
        ann_id, coco_name, ex_dark_name = label.strip().split(',')[0], label.strip().split(',')[2], label.strip().split(',')[3]
        label_dict[str(ex_dark_name)] = int(ann_id)
        categories.append({'id': int(ann_id), 'name': coco_name, 'supercategory': coco_name})

    # Save the JSON for train / val / test
    for train_type in [1, 2, 3]:
        # Output Format
        coco_output = {
            "info": INFO,
            "licenses": LICENSES,
            "categories": categories,
            "images": [],
            "annotations": []
        }

        ann_paths = glob(os.path.join(exdark_dir, 'annotation', '*', '*.txt'))
        ann_paths.sort()

        image_id = 0
        annotation_id = 0

        # Go through all image file
        for _, ann_path in enumerate(tqdm(ann_paths)):
            # Load single annotation file
            with open(ann_path, 'r') as f:
                ann_sly = f.readlines()
                ann_sly = ann_sly[1:]

            # Check Training Type
            img_name = ann_path.replace('annotation', 'image').replace('.txt', '').split('/')[-1]

            # Check Image Format (.jpg, .png, .JPEG, ...) => there are some errors in annotation files
            img_name = '%s.%s' % (img_name.split('.')[0], file_exe_dict[img_name.split('.')[0]])

            # Select Proper Train Type
            if train_type != meta_dict[img_name]:
                continue

            # Get Image Information
            image_info = create_image_info(ann_path, img_name, image_id)
            coco_output["images"].append(image_info)

            # Load annotated label in a single annotation file
            for annotation in ann_sly:
                annotation = annotation.strip().split(' ')

                # Get class ID
                ann_class = annotation[0]
                category_id = label_dict[ann_class]

                # Get Annotation
                x, y, w, h = list(map(int, annotation[1:5]))
                area = w * h
                segmentation = [[x, y, x+w, y, x+w, y+h, x, y+h]]
                bbox = [x, y, w, h]

                if category_id is not None:
                    annotation = {
                        'iscrowd': 0,
                        'image_id': image_id,
                        'segmentation': segmentation,
                        'category_id': category_id,
                        'id': annotation_id,
                        'bbox': bbox,
                        'area': area
                    }

                    coco_output["annotations"].append(annotation)
                    annotation_id += 1

                else:
                    continue

            image_id += 1

        # write final annotation file
        if train_type == 1:
            type = 'train'
        elif train_type == 2:
            type = 'val'
        elif train_type == 3:
            type = 'test'
        else:
            raise('Not proper Train Type')

        save_path = os.path.join(save_folder, 'annotations', 'instances_%s2017.json' %type)
        with open(save_path, 'w') as f:
            json.dump(coco_output, f)
        print('saved')


if __name__=='__main__':
    # split_train_val_exdark(exdark_dir='/data/sung/dataset/low-illumination-dataset/exdark',
    #                        exdark_meta_path='/data/sung/dataset/low-illumination-dataset/exdark/exdark_meta.txt',
    #                        save_folder='/data/sung/dataset/low-illumination-dataset/coco_exdark')

    exdark_coco_converter(exdark_dir='/data/sung/dataset/low-illumination-dataset/exdark',
                          exdark_label_path='/data/sung/dataset/low-illumination-dataset/exdark/exdark_labels.txt',
                          exdark_meta_path='/data/sung/dataset/low-illumination-dataset/exdark/exdark_meta.txt',
                          save_folder='/data/sung/dataset/low-illumination-dataset/coco_exdark/')



