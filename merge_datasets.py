import os
import shutil
from pathlib import Path

UNIFIED_CLASSES = ['person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck', 'auto_rickshaw', 'animal', 'pothole', 'traffic_cone', 'traffic_light', 'traffic_sign']

COCO_MAPPING = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}

IDD_MAPPING = {0: 0, 1: 0, 2: 2, 3: 4, 4: 5, 5: 2, 6: 3, 7: 1, 8: 6, 9: 7, 10: 10, 11: 11}

def remap_labels(input_labels_dir, output_labels_dir, mapping):
    os.makedirs(output_labels_dir, exist_ok=True)
    for label_file in Path(input_labels_dir).glob("*.txt"):
        with open(label_file, 'r') as f:
            lines = f.readlines()
        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if parts:
                old_cls = int(parts[0])
                if old_cls in mapping:
                    parts[0] = str(mapping[old_cls])
                    new_lines.append(' '.join(parts) + '\n')
        if new_lines:
            with open(f"{output_labels_dir}/{label_file.name}", 'w') as f:
                f.writelines(new_lines)

def copy_images(input_images_dir, output_images_dir, output_labels_dir):
    os.makedirs(output_images_dir, exist_ok=True)
    for label_file in Path(output_labels_dir).glob("*.txt"):
        img_name = label_file.stem + '.jpg'
        img_path = f"{input_images_dir}/{img_name}"
        if os.path.exists(img_path):
            shutil.copy(img_path, f"{output_images_dir}/{img_name}")

merged_dir = 'merged_coco_idd'

# Train merge
remap_labels('datasets/COCO/labels/train', f'{merged_dir}/labels/train', COCO_MAPPING)
copy_images('datasets/COCO/train2017', f'{merged_dir}/images/train', f'{merged_dir}/labels/train')
remap_labels('datasets/IDD_YOLO/train/labels', f'{merged_dir}/labels/train', IDD_MAPPING)
copy_images('datasets/IDD_YOLO/train/images', f'{merged_dir}/images/train', f'{merged_dir}/labels/train')

# Val merge
remap_labels('datasets/COCO/labels/val', f'{merged_dir}/labels/val', COCO_MAPPING)
copy_images('datasets/COCO/val2017', f'{merged_dir}/images/val', f'{merged_dir}/labels/val')
remap_labels('datasets/IDD_YOLO/val/labels', f'{merged_dir}/labels/val', IDD_MAPPING)
copy_images('datasets/IDD_YOLO/val/images', f'{merged_dir}/images/val', f'{merged_dir}/labels/val')