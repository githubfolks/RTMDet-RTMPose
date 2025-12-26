import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
import os
import glob

class YOLODataset(Dataset):
    """
    Dataset for YOLO format (Txt files).
    Supports both Detection and Pose if keypoints are present.
    
    Format:
    <class> <cx> <cy> <w> <h> <p1x> <p1y> <p1v> ...
    Normalized coordinates (0-1).
    """
    def __init__(self, data_root, split='train', num_keypoints=17, use_pose=False, transform=None, cache_ram=False):
        self.data_root = data_root
        self.split = split
        self.use_pose = use_pose
        self.num_keypoints = num_keypoints
        self.transform = transform
        self.cache_ram = cache_ram
        
        # Structure: data_root/images/split/*.jpg and data_root/labels/split/*.txt
        self.img_dir = os.path.join(data_root, 'images', split)
        self.lbl_dir = os.path.join(data_root, 'labels', split)
        
        self.img_files = sorted(glob.glob(os.path.join(self.img_dir, '*.*')))
        # Filter for valid images
        valid_exts = {'.jpg', '.jpeg', '.png', '.bmp'}
        self.img_files = [f for f in self.img_files if os.path.splitext(f)[1].lower() in valid_exts]
        
        self.imgs = [None] * len(self.img_files)
        if self.cache_ram:
            print(f"Loading {len(self.img_files)} images into RAM...")
            for i, img_path in enumerate(self.img_files):
                img = cv2.imread(img_path)
                if img is not None:
                    self.imgs[i] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            print("RAM Caching Complete.")
        
    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        
        # Infer label path
        # images/train/foo.jpg -> labels/train/foo.txt
        # Handle simple replacement
        lbl_path = os.path.join(self.lbl_dir, os.path.basename(img_path).rsplit('.', 1)[0] + '.txt')
        
        if self.cache_ram and self.imgs[idx] is not None:
            img = self.imgs[idx].copy() # Copy to avoid mutation in transform
        else:
            img = cv2.imread(img_path)
            if img is None:
                raise FileNotFoundError(f"Image not found: {img_path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # H, W, 3
        
        h, w, _ = img.shape
        
        bboxes = []
        labels = []
        keypoints = []
        
        if os.path.exists(lbl_path):
            with open(lbl_path, 'r') as f:
                lines = f.readlines()
                
            for line in lines:
                parts = list(map(float, line.strip().split()))
                cls_id = int(parts[0])
                cx, cy, bw, bh = parts[1:5]
                
                # Denormalize Box
                x1 = (cx - bw/2) * w
                y1 = (cy - bh/2) * h
                x2 = (cx + bw/2) * w
                y2 = (cy + bh/2) * h
                
                bboxes.append([x1, y1, x2, y2])
                labels.append(cls_id)
                
                if self.use_pose:
                    # Keypoints start at index 5
                    # Groups of 3: x, y, v
                    kpts = parts[5:]
                    num_kpts_in_file = len(kpts) // 3
                    
                    # Pad or truncate to self.num_keypoints if mismatch (warning needed?)
                    # Assuming match
                    current_kpts = []
                    for i in range(self.num_keypoints):
                        if i < num_kpts_in_file:
                            kx = kpts[i*3] * w
                            ky = kpts[i*3+1] * h
                            kv = int(kpts[i*3+2]) # 2=visible, 1=occluded, 0=absent usually
                        else:
                            kx, ky, kv = 0, 0, 0
                        current_kpts.append([kx, ky, float(kv)])
                    keypoints.append(current_kpts)
        
        target = {
            'bboxes': torch.tensor(bboxes, dtype=torch.float32),
            'labels': torch.tensor(labels, dtype=torch.int64)
        }
        
        if self.use_pose:
            target['keypoints'] = torch.tensor(keypoints, dtype=torch.float32)
        
        if self.transform:
            img, target = self.transform(img, target)
            
        return img, target
