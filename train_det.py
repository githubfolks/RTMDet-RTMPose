import torch


import torch.optim as optim
from torch.utils.data import DataLoader
from models.rtmdet import RTMDet

from data.coco_det import COCODetDataset
import argparse
import os
import cv2
import numpy as np
import datetime
import yaml




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
        
    # Collate function defined globally now
        
    num_workers = item_dict.get('num_workers', 0)
    persistent_workers = item_dict.get('persistent_workers', False)
    
    # On Windows, persistent_workers=True with num_workers>0 can sometimes cause hangs
    if num_workers == 0:
        persistent_workers = False
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, 
                            num_workers=num_workers, pin_memory=True, persistent_workers=persistent_workers)
    
    # Model
    model = RTMDet('s').to(device)
    
    model = RTMDet('s').to(device)
            
    model.train()
    

    # Lower learning rate to prevent gradient explosion on MPS
    base_lr = item_dict.get('lr', 1e-5)
    optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=0.05)
    
    start_epoch = 0
    
    if resume_path and os.path.exists(resume_path):
        try:
             checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
             
             # 1. Load Model Weights
             if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                 model.load_state_dict(checkpoint['state_dict'])
                 print("Resumed model weights from dict checkpoint.")
             else:
                 # Fallback for old/plain weights
                 model.load_state_dict(checkpoint)
             
             # 2. Load Optimizer (if dict)
             if isinstance(checkpoint, dict) and 'optimizer' in checkpoint:
                 optimizer.load_state_dict(checkpoint['optimizer'])
                 print("Resumed optimizer state.")
             
             # 3. Load Epoch (if dict)
             if isinstance(checkpoint, dict) and 'epoch' in checkpoint:
                 start_epoch = checkpoint['epoch']
                 print(f"Resuming training from epoch {start_epoch + 1}.")
                 
        except Exception as e:
             print(f"Warning: Issue parsing checkpoint for resume: {e}")
             print("Falling back to random init (or partial load) and starting from epoch 0.")
             start_epoch = 0

    # If not resuming, check for load_from (pretrained weights)
    if not (resume_path and os.path.exists(resume_path)):
        load_from = item_dict.get('load_from', None)
        if load_from and os.path.exists(load_from):
            print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Loading pretrained weights from {load_from}...")
            try:
                checkpoint = torch.load(load_from, map_location=device, weights_only=False)
                if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                     model.load_state_dict(checkpoint['state_dict'], strict=False)
                     print("Loaded model weights from dict checkpoint.")
                else:
                     model.load_state_dict(checkpoint, strict=False)
                     print("Loaded model weights from weights-only checkpoint.")
            except Exception as e:
                print(f"Warning: Failed to load pretrained weights: {e}")

    # Learning rate warmup
    warmup_steps = 500
    
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Starting training from epoch {start_epoch + 1}...")
    
    # Calculate global step based on resumed epoch
    global_step = start_epoch * len(dataloader)
    
    for epoch in range(start_epoch, epochs):
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
            
            # Cosine Annealing (Manual implementation to work with per-step warmup)
            # Or use torch.optim.lr_scheduler.CosineAnnealingLR at epoch level?
            # Let's use simple cosine decay at step level after warmup:
            else:
                progress = (global_step - warmup_steps) / (epochs * len(dataloader) - warmup_steps)
                progress = min(progress, 1.0)
                cosine_lr = base_lr * 0.5 * (1 + np.cos(np.pi * progress))
                for param_group in optimizer.param_groups:
                    param_group['lr'] = cosine_lr
            
            optimizer.zero_grad()
            
            cls_scores, bbox_preds = model(imgs) # Forward
            
            loss_dict = model.head.loss(cls_scores, bbox_preds, gt_bboxes, gt_labels)
            

            loss = loss_dict['loss_cls'] + loss_dict['loss_bbox']
            loss = torch.clamp(loss, min=0.0, max=15.0)

            # Check for NaN loss and skip batch if detected
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Warning: NaN/Inf loss detected at step {i}, skipping batch...")
                optimizer.zero_grad()
                global_step += 1
                continue
            
            loss.backward()
            
            # 🔒 Sanitize gradients
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.data = torch.nan_to_num(p.grad.data, nan=0.0, posinf=0.0, neginf=0.0)

            # Gradient Clipping - more aggressive for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            
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
    parser.add_argument('--config', type=str, default=None, help='Path to YAML config file')
    parser.add_argument('--data_root', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=None) 
    parser.add_argument('--epochs', type=int, default=None) 
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--work_dir', type=str, default=None, help='Directory to save weights')
    parser.add_argument('--device', type=str, default=None, help='Device to use (auto, cuda, mps, cpu)')
    parser.add_argument('--lr', type=float, default=None, help='Learning rate')
    args = parser.parse_args()
    
    cfg_args = {}
    
    # Defaults
    final_args = {
        'data_root': 'dataset_coco', # Default if nothing else specifies
        'batch_size': 16,
        'epochs': 100,
        'resume': None,
        'work_dir': 'train/weights_scratch',
        'device': 'auto',
        'lr': 5e-5
    }

    if args.config:
        with open(args.config, 'r') as f:
            cfg = yaml.safe_load(f)
            if 'detection' in cfg:
                det_cfg = cfg['detection']
                
                # Map config to args
                if 'data' in det_cfg:
                    if 'root' in det_cfg['data']:
                        cfg_args['data_root'] = det_cfg['data']['root']
                    if 'batch_size' in det_cfg['data']:
                        cfg_args['batch_size'] = det_cfg['data']['batch_size']
                
                if 'training' in det_cfg:
                    if 'epochs' in det_cfg['training']:
                        cfg_args['epochs'] = det_cfg['training']['epochs']
                    if 'device' in det_cfg['training']:
                        cfg_args['device'] = det_cfg['training']['device']
                    if 'work_dir' in det_cfg['training']:
                        cfg_args['work_dir'] = det_cfg['training']['work_dir']
                    if 'resume' in det_cfg['training']:
                        cfg_args['resume'] = det_cfg['training']['resume']
                    if 'load_from' in det_cfg['training']:
                        cfg_args['load_from'] = det_cfg['training']['load_from']
                
                if 'optimization' in det_cfg and 'learning_rate' in det_cfg['optimization']:
                     if 'base_lr' in det_cfg['optimization']['learning_rate']:
                         cfg_args['lr'] = det_cfg['optimization']['learning_rate']['base_lr']

    # Update defaults with config values
    final_args.update(cfg_args)
    
    # Override with command line args if they are not None
    cmd_args = {k: v for k, v in vars(args).items() if v is not None}
    final_args.update(cmd_args)
    
    # Check required
    if not final_args.get('data_root'):
         raise ValueError("data_root must be specified either in config or via command line")

    # Clean up non-training keys if any (like 'config')
    if 'config' in final_args:
        del final_args['config']
        
    print(f"Training with arguments: {final_args}")
    train(final_args)
