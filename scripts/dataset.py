import os
import json
import pickle
import torch
from torch.utils.data import Dataset
from PIL import Image

class RefCOCODataset(Dataset):
    def __init__(self, image_dir, annotation_file, refexp_file, transform=None):
        self.image_dir = image_dir
        self.transform = transform

        with open(annotation_file, 'r') as f:
            self.coco_data = json.load(f)

        with open(refexp_file, 'rb') as f:
            self.referring_expressions = pickle.load(f)

        self.image_id_to_info = {img['id']: img for img in self.coco_data['images']}
        self.ann_id_to_info = {ann['id']: ann for ann in self.coco_data['annotations']}

        self.data = []
        for ref in self.referring_expressions:
            ann = self.ann_id_to_info[ref['ann_id']]
            img_info = self.image_id_to_info[ref['image_id']]
            for sentence in ref['sentences']:
                self.data.append({
                    'image_id': img_info['id'],
                    'file_name': img_info['file_name'],
                    'bbox': ann['bbox'],
                    'caption': sentence['raw']
                })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        img_path = os.path.join(self.image_dir, sample['file_name'])
        image = Image.open(img_path).convert('RGB')

        bbox = sample['bbox']
        original_width, original_height = image.size
        x_min, y_min, width, height = bbox
        x_min = (x_min / original_width) * 224
        y_min = (y_min / original_height) * 224
        width = (width / original_width) * 224
        height = (height / original_height) * 224
        scaled_bbox = [x_min, y_min, x_min + width, y_min + height]

        if self.transform:
            image = self.transform(image)

        return {
            'image': image,
            'bbox': torch.tensor(scaled_bbox, dtype=torch.float32),
            'caption': sample['caption']
        }