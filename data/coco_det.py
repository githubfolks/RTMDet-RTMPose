import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
import os
import json

class COCODetDataset(Dataset):
    """
    COCO Dataset for Object Detection.
    """
    def __init__(self, data_root, ann_file, transform=None):
        self.data_root = data_root
        self.transform = transform
        
        # Load annotations
        with open(ann_file, 'r') as f:
            self.coco = json.load(f)
            
        self.images = {img['id']: img for img in self.coco['images']}
        self.annotations = {}
        for ann in self.coco['annotations']:
            img_id = ann['image_id']
            if img_id not in self.annotations:
                self.annotations[img_id] = []
            self.annotations[img_id].append(ann)
            
        self.img_ids = list(self.images.keys())

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_info = self.images[img_id]
        file_name = img_info['file_name']
        img_path = os.path.join(self.data_root, file_name)
        
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        anns = self.annotations.get(img_id, [])
        bboxes = []
        labels = []
        
        for ann in anns:
            x, y, w, h = ann['bbox']
            bboxes.append([x, y, x+w, y+h])
            labels.append(ann['category_id'])
            
        target = {
            'bboxes': torch.tensor(bboxes, dtype=torch.float32),
            'labels': torch.tensor(labels, dtype=torch.int64)
        }
        
        if self.transform:
            img, target = self.transform(img, target)
            
        return img, target
