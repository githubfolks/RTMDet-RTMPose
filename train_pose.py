import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from models.rtmpose import RTMPose

from data.coco_pose import COCOPoseDataset
import argparse
import torch.nn.functional as F
import datetime
import os

import cv2
import numpy as np

# Transform moved to global scope for multiprocessing
class CropPersonTransform:
    """
    Top-Down approach: Crop each person using their bounding box,
    then resize to target size. Keypoints are transformed to crop-relative coordinates.
    """
    def __init__(self, size=(192, 256), padding_ratio=0.25):
        self.size = size  # W, H
        self.padding_ratio = padding_ratio
        
    def __call__(self, img, target):
        """
        Returns a list of (cropped_img, cropped_keypoints) for each person.
        """
        h_img, w_img, _ = img.shape
        target_w, target_h = self.size
        
        bboxes = target['bboxes']
        keypoints = target['keypoints']
        
        cropped_samples = []
        
        for i in range(len(bboxes)):
            bbox = bboxes[i].numpy() if isinstance(bboxes[i], torch.Tensor) else bboxes[i]
            kpts = keypoints[i].numpy() if isinstance(keypoints[i], torch.Tensor) else keypoints[i]
            
            x1, y1, x2, y2 = bbox
            w, h = x2 - x1, y2 - y1
            
            # Skip invalid boxes
            if w <= 0 or h <= 0:
                continue
            
            # Add padding to the bounding box
            pad_w = w * self.padding_ratio
            pad_h = h * self.padding_ratio
            
            x1_pad = max(0, int(x1 - pad_w))
            y1_pad = max(0, int(y1 - pad_h))
            x2_pad = min(w_img, int(x2 + pad_w))
            y2_pad = min(h_img, int(y2 + pad_h))
            
            # Crop the person
            crop = img[y1_pad:y2_pad, x1_pad:x2_pad]
            crop_h, crop_w = crop.shape[:2]
            
            if crop_w < 10 or crop_h < 10:
                continue
            
            # Resize crop to target size
            crop_resized = cv2.resize(crop, (target_w, target_h))
            
            # Transform keypoints to crop-relative coordinates
            scale_x = target_w / crop_w
            scale_y = target_h / crop_h
            
            transformed_kpts = []
            for kpt in kpts:
                kx, ky, kv = kpt
                if kv > 0:  # Only transform visible keypoints
                    # Shift to crop origin, then scale to target size
                    new_kx = (kx - x1_pad) * scale_x
                    new_ky = (ky - y1_pad) * scale_y
                    # Clamp to valid range
                    new_kx = max(0, min(target_w - 1, new_kx))
                    new_ky = max(0, min(target_h - 1, new_ky))
                else:
                    new_kx, new_ky = 0, 0
                transformed_kpts.append([new_kx, new_ky, kv])
            
            cropped_samples.append((crop_resized, np.array(transformed_kpts)))
        
        return cropped_samples

def collate_fn(batch):
    """
    Collate function for Top-Down pose training.
    Each item in batch is a list of (crop, keypoints) tuples.
    We flatten all person crops into a single batch.
    """
    imgs = []
    kpts_targets = []
    
    for samples in batch:
        # samples is a list of (crop, keypoints) for each person in the image
        for crop, kpts in samples:
            imgs.append(torch.from_numpy(crop).permute(2, 0, 1).float() / 255.0)
            kpts_targets.append(torch.tensor(kpts, dtype=torch.float32))
    
    if len(imgs) == 0:
        # Return empty batch if no valid samples
        return torch.zeros(1, 3, 256, 192), torch.zeros(1, 17, 3)
    
    return torch.stack(imgs), torch.stack(kpts_targets)


def train(item_dict):
    data_root = item_dict['data_root']
    epochs = item_dict['epochs']
    batch_size = item_dict['batch_size']
    resume_path = item_dict.get('resume', None)
    work_dir = item_dict.get('work_dir', 'train/weights')
    
    os.makedirs(work_dir, exist_ok=True)

    # Setup logging
    log_file = os.path.join(work_dir, 'training_pose_log.txt')
    def log(msg):
        print(msg)
        with open(log_file, 'a') as f:
            f.write(msg + '\n')

    # Device selection: CUDA > MPS (Mac) > CPU
    # ... (existing device logic is fine, but lets update prints to use log()) ...
    if torch.cuda.is_available():
        device = torch.device('cuda')
        log(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Training on device: {device}")
        log(f"GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        log(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Training on device: {device} (Apple Silicon GPU)")
    else:
        device = torch.device('cpu')
        log(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Training on device: {device}")
    
    # Dataset
    img_dir = os.path.join(data_root, 'train', 'images')
    ann_file = os.path.join(data_root, 'train', 'train.json')
    
    # Transform
    # Pass transform to dataset
    dataset = COCOPoseDataset(img_dir, ann_file, transform=CropPersonTransform(size=(192, 256), padding_ratio=0.25))
    
    num_workers = 4
    persistent_workers = True

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn,
                            num_workers=num_workers, pin_memory=True, persistent_workers=persistent_workers)
    
    # Model
    model = RTMPose('s', input_size=(256, 192)).to(device)
    
    if resume_path and os.path.exists(resume_path):
        log(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Resuming from {resume_path}...")
        try:
            checkpoint = torch.load(resume_path, map_location=device)
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                 model.load_state_dict(checkpoint['state_dict'])
                 log("Resumed model weights from dict checkpoint.")
            else:
                 model.load_state_dict(checkpoint)
                 log("Resumed from weights-only checkpoint.")
        except Exception as e:
            log(f"Warning: Failed to resume: {e}")
            
    model.train()
    
    # Learning rate from args
    lr = item_dict.get('lr', 0.001)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    
    start_epoch = 0
    if resume_path and os.path.exists(resume_path):
        try:
            checkpoint = torch.load(resume_path, map_location=device)
            if isinstance(checkpoint, dict):
                if 'optimizer' in checkpoint:
                     optimizer.load_state_dict(checkpoint['optimizer'])
                     log("Resumed optimizer state.")
                if 'epoch' in checkpoint:
                     start_epoch = checkpoint['epoch']
                     log(f"Resuming training from epoch {start_epoch + 1}.")
        except:
            pass
    
    log(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Starting Pose Training from epoch {start_epoch + 1}...")
    for epoch in range(start_epoch, epochs):
        total_loss = 0
        for i, (imgs, gt_kpts) in enumerate(dataloader):
            imgs = imgs.to(device)
            gt_kpts = gt_kpts.to(device)
            
            optimizer.zero_grad()
            
            # Forward
            pred_x, pred_y = model(imgs)
            
            loss = simcc_loss(pred_x, pred_y, gt_kpts)
            
            loss.backward()
            
            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            
            if i % 10 == 0:
                print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Epoch [{epoch+1}/{epochs}], Step [{i}/{len(dataloader)}], Loss: {loss.item():.4f}")
                
        avg_loss = total_loss/len(dataloader)
        log(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Epoch {epoch+1} Complete. Avg Loss: {avg_loss:.4f}")
        
        # Save every epoch
        ckpt_name = f'rtmpose_custom_epoch_{epoch+1}.pth'
        save_path = os.path.join(work_dir, ckpt_name)
        checkpoint = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'loss': avg_loss
        }
        torch.save(checkpoint, save_path)
        
        # Latest
        torch.save(checkpoint, os.path.join(work_dir, 'rtmpose_custom.pth'))
        
    log(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Pose Training Complete. Model saved to {save_path}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=50) # Increased to 50
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--work_dir', type=str, default='train/weights', help='Directory to save weights')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    args = parser.parse_args()
    
    train(vars(args))
