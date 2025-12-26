import torch
print(f"DEBUG: Top of script. CUDA Available? {torch.cuda.is_available()}", flush=True)

import torch.optim as optim
from torch.utils.data import DataLoader
from models.rtmdet import RTMDet
from data.yolo import YOLODataset
import argparse
import os
import cv2
import numpy as np
import datetime

print(f"DEBUG: After imports. CUDA Available? {torch.cuda.is_available()}", flush=True)


# Transform moved to global scope for multiprocessing
class ResizeTransform:
    def __init__(self, size=(640, 640)): # W, H
        self.size = size
        
    def __call__(self, img, target):
        # img: H, W, 3
        h, w, _ = img.shape
        target_w, target_h = self.size
        
        img_resized = cv2.resize(img, (target_w, target_h))
        
        # Scale bboxes
        scale_x = target_w / w
        scale_y = target_h / h
        
        if 'bboxes' in target and len(target['bboxes']) > 0:
            bboxes = target['bboxes']
            # bboxes: (N, 4) -> x1, y1, x2, y2
            bboxes[:, 0] *= scale_x
            bboxes[:, 1] *= scale_y
            bboxes[:, 2] *= scale_x
            bboxes[:, 3] *= scale_y
            target['bboxes'] = bboxes
        
        return img_resized, target

def collate_fn(batch):
    imgs = []
    bboxes = []
    labels = []
    for img, target in batch:
        imgs.append(torch.from_numpy(img).permute(2, 0, 1).float() / 255.0)
        bboxes.append(target['bboxes'])
        labels.append(target['labels'])
    return torch.stack(imgs), bboxes, labels

def train(item_dict):
    data_root = item_dict['data_root']
    epochs = item_dict['epochs']
    batch_size = item_dict['batch_size']
    resume_path = item_dict.get('resume', None)
    work_dir = item_dict.get('work_dir', 'train/weights')
    
    os.makedirs(work_dir, exist_ok=True)

    
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not detected. Please check your environment or drivers.")
    
    device = torch.device('cuda')
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Training on device: {device}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Transform
    # Transform defined globally now

    # Dataset
    # Assuming standard structure: data_root/images/train
    dataset = YOLODataset(data_root, split='train', use_pose=False, transform=ResizeTransform(size=(640, 640)), cache_ram=True)
    
    # Collate function to handle list of tensors
    # Collate function defined globally now
        
    # When using RAM Cache on Windows, num_workers must be 0 to avoid expensive pickling of the huge dataset object
    num_workers = 0 if dataset.cache_ram else 4
    persistent_workers = False if num_workers == 0 else True
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, 
                            num_workers=num_workers, pin_memory=True, persistent_workers=persistent_workers)
    
    # Model
    model = RTMDet('s').to(device)
    
    if resume_path and os.path.exists(resume_path):
        print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Resuming from {resume_path}...")
        try:
            model.load_state_dict(torch.load(resume_path, map_location=device))
        except Exception as e:
            print(f"Warning: Failed to resume: {e}")
            
    model.train()
    
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.05)
    
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Starting training...")
    for epoch in range(epochs):
        total_loss = 0
        for i, (imgs, gt_bboxes, gt_labels) in enumerate(dataloader):
            imgs = imgs.to(device)
            # gt_bboxes/labels are lists of tensors, move them to device when needed in loss
            gt_bboxes = [b.to(device) for b in gt_bboxes]
            gt_labels = [l.to(device) for l in gt_labels]
            
            optimizer.zero_grad()
            
            cls_scores, bbox_preds = model(imgs) # Forward
            
            loss_dict = model.head.loss(cls_scores, bbox_preds, gt_bboxes, gt_labels)
            
            loss = loss_dict['loss_cls'] + loss_dict['loss_bbox']
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if i % 10 == 0:
                print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Epoch [{epoch+1}/{epochs}], Step [{i}/{len(dataloader)}], Loss: {loss.item():.4f}")
                
        print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Epoch {epoch+1} Complete. Avg Loss: {total_loss/len(dataloader):.4f}")
        
        # Save every epoch
        save_path = os.path.join(work_dir, 'rtmdet_custom.pth')
        torch.save(model.state_dict(), save_path)
        
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Training Complete. Model saved to {save_path}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=50) # Increased to 50
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--work_dir', type=str, default='train/weights', help='Directory to save weights')
    args = parser.parse_args()
    
    train(vars(args))
