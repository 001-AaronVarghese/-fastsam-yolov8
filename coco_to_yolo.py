import json
import os
from tqdm import tqdm

def convert_coco_to_yolo(json_path, images_dir, output_labels_dir, target_classes):
    with open(json_path) as f:
        data = json.load(f)
    os.makedirs(output_labels_dir, exist_ok=True)
    category_map = {cat['id']: i for i, cat in enumerate(data['categories']) if cat['id'] in target_classes}
    anns_by_img = {}
    for ann in data['annotations']:
        if ann['category_id'] in target_classes:
            anns_by_img.setdefault(ann['image_id'], []).append(ann)
    for img in tqdm(data['images']):
        img_id = img['id']
        if img_id in anns_by_img:
            label_path = f"{output_labels_dir}/{img['file_name'].replace('.jpg', '.txt')}"
            with open(label_path, 'w') as f:
                for ann in anns_by_img[img_id]:
                    cls_id = category_map[ann['category_id']]
                    x, y, w, h = ann['bbox']
                    x_center = (x + w / 2) / img['width']
                    y_center = (y + h / 2) / img['height']
                    w_norm = w / img['width']
                    h_norm = h / img['height']
                    f.write(f"{cls_id} {x_center} {y_center} {w_norm} {h_norm}\n")

TARGET_CLASSES = [1, 2, 3, 4, 6, 8]
convert_coco_to_yolo('datasets/COCO/annotations/instances_train2017.json', 'datasets/COCO/train2017', 'datasets/COCO/labels/train', TARGET_CLASSES)
convert_coco_to_yolo('datasets/COCO/annotations/instances_val2017.json', 'datasets/COCO/val2017', 'datasets/COCO/labels/val', TARGET_CLASSES)