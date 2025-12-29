import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
import os
import json

class COCOPoseDataset(Dataset):
    """
    COCO Dataset for Pose Estimation.
    """
    def __init__(self, data_root, ann_file, transform=None):
        self.data_root = data_root
        self.transform = transform
        
        with open(ann_file, 'r') as f:
            self.coco = json.load(f)
            
        self.images = {img['id']: img for img in self.coco['images']}
        self.annotations = {}
        # Filter for keypoints
        for ann in self.coco['annotations']:
            if 'keypoints' in ann and ann['num_keypoints'] > 0:
                img_id = ann['image_id']
                if img_id not in self.annotations:
                    self.annotations[img_id] = []
                self.annotations[img_id].append(ann)
        
        # Only keep images with keypoint annotations
        self.img_ids = list(self.annotations.keys())

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_info = self.images[img_id]
        file_name = img_info['file_name']
        img_path = os.path.join(self.data_root, file_name)
        
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        anns = self.annotations[img_id]
        
        # For top-down pose, we usually crop each person.
        # This dataset loader might return the full image + list of persons, 
        # or simplified to return a specific person.
        # Simplified: Return full image + all keypoints.
        
        keypoints = []
        bboxes = []
        for ann in anns:
            kps = np.array(ann['keypoints']).reshape(-1, 3) # x, y, v
            keypoints.append(kps)
            x, y, w, h = ann['bbox']
            bboxes.append([x, y, x+w, y+h])
            
        target = {
            'keypoints': torch.tensor(keypoints, dtype=torch.float32),
            'bboxes': torch.tensor(bboxes, dtype=torch.float32)
        }
        

        if self.transform:
            res = self.transform(img, target)
            if isinstance(res, list):
                return res
            img, target = res
            
        return img, target
