import torch
print(f"DEBUG: Top of script. CUDA Available? {torch.cuda.is_available()}", flush=True)

import torch.optim as optim
from torch.utils.data import DataLoader
from models.rtmdet import RTMDet

from data.coco_det import COCODetDataset
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
    
    # Setup logging
    log_file = os.path.join(work_dir, 'training_det_log.txt')
    def log(msg):
        print(msg)
        with open(log_file, 'a') as f:
            f.write(msg + '\n')

    
    requested_device = item_dict.get('device', 'auto')
    
    if requested_device != 'auto':
        device = torch.device(requested_device)
        print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Training on user-requested device: {device}")
    else:
        # Device selection: CUDA > MPS (Mac) > CPU
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Training on device: {device}")
            print(f"GPU: {torch.cuda.get_device_name(0)}")

        elif torch.backends.mps.is_available():
            # RTMDet is currently unstable on MPS (NaN loss). Defaulting to CPU for safety.
            print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] MPS detected but unstable for RTMDet. Defaulting to CPU.")
            device = torch.device('cpu')
        else:
            device = torch.device('cpu')
            print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Training on device: {device}")
    
    # Transform
    # Transform defined globally now

    # Dataset
    # COCO Format
    img_dir = os.path.join(data_root, 'train', 'images')
    ann_file = os.path.join(data_root, 'train', 'train.json')
    
    dataset = COCODetDataset(img_dir, ann_file, transform=ResizeTransform(size=(640, 640)))
    

    # Collate function to handle list of tensors
    # Collate function defined globally now
        
    num_workers = 4
    persistent_workers = True
    
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
    

    # Lower learning rate to prevent gradient explosion on MPS
    base_lr = item_dict.get('lr', 1e-5)
    optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=0.05)
    
    # Learning rate warmup
    warmup_steps = 500
    
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Starting training...")
    global_step = 0
    for epoch in range(epochs):
        total_loss = 0
        valid_steps = 0
        for i, (imgs, gt_bboxes, gt_labels) in enumerate(dataloader):
            imgs = imgs.to(device)
            # gt_bboxes/labels are lists of tensors, move them to device when needed in loss
            gt_bboxes = [b.to(device) for b in gt_bboxes]
            gt_labels = [l.to(device) for l in gt_labels]
            
            # Warmup learning rate
            if global_step < warmup_steps:
                warmup_lr = base_lr * (global_step + 1) / warmup_steps
                for param_group in optimizer.param_groups:
                    param_group['lr'] = warmup_lr
            
            optimizer.zero_grad()
            
            cls_scores, bbox_preds = model(imgs) # Forward
            
            loss_dict = model.head.loss(cls_scores, bbox_preds, gt_bboxes, gt_labels)
            

            loss = loss_dict['loss_cls'] + loss_dict['loss_bbox']
            
            # Check for NaN loss and skip batch if detected
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Warning: NaN/Inf loss detected at step {i}, skipping batch...")
                optimizer.zero_grad()
                global_step += 1
                continue
            
            loss.backward()
            
            # Replace NaN/Inf gradients with zeros to allow partial learning
            nan_grad_count = 0
            for param in model.parameters():
                if param.grad is not None:
                    nan_mask = torch.isnan(param.grad) | torch.isinf(param.grad)
                    if nan_mask.any():
                        param.grad = torch.where(nan_mask, torch.zeros_like(param.grad), param.grad)
                        nan_grad_count += 1
            
            if nan_grad_count > 0 and i % 100 == 0:
                print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Fixed {nan_grad_count} NaN gradients at step {i}...")
            
            # Gradient Clipping - more aggressive for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            valid_steps += 1
            global_step += 1
            
            if i % 10 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Epoch [{epoch+1}/{epochs}], Step [{i}/{len(dataloader)}], Loss: {loss.item():.4f}, LR: {current_lr:.2e}")
                
        avg_loss = total_loss/max(valid_steps, 1)
        log(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Epoch {epoch+1} Complete. Avg Loss: {avg_loss:.4f}")
        

        # Save every epoch
        # Save checkpoint with epoch number
        ckpt_name = f'rtmdet_custom_epoch_{epoch+1}.pth'
        save_path = os.path.join(work_dir, ckpt_name)
        checkpoint = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'loss': avg_loss
        }
        torch.save(checkpoint, save_path)
        
        # Also update the 'latest' file for easy resumption
        latest_path = os.path.join(work_dir, 'rtmdet_custom.pth')
        torch.save(checkpoint, latest_path)
        
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Training Complete. Model saved to {save_path}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=100) # Increased for scratch training
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--work_dir', type=str, default='train/weights_scratch', help='Directory to save weights')
    parser.add_argument('--device', type=str, default='auto', help='Device to use (auto, cuda, mps, cpu)')
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate')
    args = parser.parse_args()
    
    train(vars(args))
