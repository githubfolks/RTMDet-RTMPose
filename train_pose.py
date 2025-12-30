import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from models.rtmpose import RTMPose

from data.coco_pose import COCOPoseDataset
import argparse
import torch.nn.functional as F
import datetime
import os
import yaml

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

# ============================================
# LOSS FUNCTION
# ============================================

def simcc_loss(pred_simcc_x, pred_simcc_y, gt_kpts, simcc_split_ratio=2.0, sigma=6.0, use_soft_label=True):
    """
    SimCC loss (copied from train_pose_improved.py).
    """
    B, K, W_bins = pred_simcc_x.shape
    _, _, H_bins = pred_simcc_y.shape
    device = pred_simcc_x.device
    
    valid_mask = gt_kpts[..., 2] > 0
    
    if valid_mask.sum() == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)
    
    gt_x = gt_kpts[..., 0] * simcc_split_ratio
    gt_y = gt_kpts[..., 1] * simcc_split_ratio
    
    if use_soft_label:
        # Generate Gaussian soft labels
        x_indices = torch.arange(W_bins, device=device).float()
        y_indices = torch.arange(H_bins, device=device).float()
        
        # (B, K, 1) - (1, 1, W_bins) -> (B, K, W_bins)
        gt_x_exp = gt_x.unsqueeze(-1)
        gt_y_exp = gt_y.unsqueeze(-1)
        
        target_x = torch.exp(-((x_indices - gt_x_exp) ** 2) / (2 * sigma ** 2))
        target_y = torch.exp(-((y_indices - gt_y_exp) ** 2) / (2 * sigma ** 2))
        
        # Normalize to sum to 1
        target_x = target_x / (target_x.sum(dim=-1, keepdim=True) + 1e-8)
        target_y = target_y / (target_y.sum(dim=-1, keepdim=True) + 1e-8)
        
        # KL divergence loss
        log_pred_x = F.log_softmax(pred_simcc_x, dim=-1)
        log_pred_y = F.log_softmax(pred_simcc_y, dim=-1)
        
        loss_x = F.kl_div(log_pred_x, target_x, reduction='none').sum(dim=-1)
        loss_y = F.kl_div(log_pred_y, target_y, reduction='none').sum(dim=-1)
    else:
        # Simple cross entropy
        gt_x = gt_x.clamp(0, W_bins-1).long()
        gt_y = gt_y.clamp(0, H_bins-1).long()
        
        loss_x = F.cross_entropy(pred_simcc_x.permute(0, 2, 1), gt_x, reduction='none')
        loss_y = F.cross_entropy(pred_simcc_y.permute(0, 2, 1), gt_y, reduction='none')
    
    loss = (loss_x * valid_mask + loss_y * valid_mask).sum() / (valid_mask.sum() + 1e-6)
    return loss


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
    
    dataset = COCOPoseDataset(img_dir, ann_file, transform=CropPersonTransform(size=(192, 256), padding_ratio=0.25))
    
    num_workers = item_dict.get('num_workers', 0)
    persistent_workers = item_dict.get('persistent_workers', False)
    
    # On Windows, persistent_workers=True with num_workers>0 can sometimes cause hangs
    if num_workers == 0:
        persistent_workers = False

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn,
                            num_workers=num_workers, pin_memory=True, persistent_workers=persistent_workers)
    
    # Model
    model = RTMPose('s', input_size=(256, 192)).to(device)
    
    model.train()
    
    # Learning rate from args
    lr = item_dict.get('lr', 0.001)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)

    start_epoch = 0
    if resume_path and os.path.exists(resume_path):
        try:
            checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
            if isinstance(checkpoint, dict):
                if 'state_dict' in checkpoint:
                     model.load_state_dict(checkpoint['state_dict'])
                     log("Resumed model weights from dict checkpoint.")
                elif isinstance(checkpoint, dict) and not 'state_dict' in checkpoint:
                    pass # Handled below
                else:
                     model.load_state_dict(checkpoint)
                     log("Resumed from weights-only checkpoint.")
                     
                if 'optimizer' in checkpoint:
                     optimizer.load_state_dict(checkpoint['optimizer'])
                     log("Resumed optimizer state.")
                if 'epoch' in checkpoint:
                     start_epoch = checkpoint['epoch']
                     log(f"Resuming training from epoch {start_epoch + 1}.")
        except Exception as e:
            log(f"Warning: Failed to resume: {e}")
            pass
    
    # If not resuming, check for load_from (pretrained weights)
    if not (resume_path and os.path.exists(resume_path)):
        load_from = item_dict.get('load_from', None)
        if load_from and os.path.exists(load_from):
            log(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Loading pretrained weights from {load_from}...")
            try:
                checkpoint = torch.load(load_from, map_location=device, weights_only=False)
                if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                     model.load_state_dict(checkpoint['state_dict'], strict=False)
                     log("Loaded model weights from dict checkpoint.")
                else:
                     model.load_state_dict(checkpoint, strict=False)
                     log("Loaded model weights from weights-only checkpoint.")
            except Exception as e:
                log(f"Warning: Failed to load pretrained weights: {e}")
    
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
    parser.add_argument('--config', type=str, default=None, help='Path to YAML config file')
    parser.add_argument('--data_root', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=None) 
    parser.add_argument('--epochs', type=int, default=None) 
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--work_dir', type=str, default=None, help='Directory to save weights')
    parser.add_argument('--lr', type=float, default=None, help='Learning rate')
    args = parser.parse_args()
    
    cfg_args = {}
    
    # Defaults
    final_args = {
        'data_root': 'dataset_coco',
        'batch_size': 16,
        'epochs': 50,
        'resume': None,
        'work_dir': 'train/weights',
        'lr': 1e-3
    }

    if args.config:
        with open(args.config, 'r') as f:
            cfg = yaml.safe_load(f)
            if 'pose' in cfg:
                pose_cfg = cfg['pose']
                
                # Map config to args
                if 'data' in pose_cfg:
                    if 'root' in pose_cfg['data']:
                        cfg_args['data_root'] = pose_cfg['data']['root']
                    if 'batch_size' in pose_cfg['data']:
                        cfg_args['batch_size'] = pose_cfg['data']['batch_size']
                
                if 'training' in pose_cfg:
                    if 'epochs' in pose_cfg['training']:
                        cfg_args['epochs'] = pose_cfg['training']['epochs']
                    if 'work_dir' in pose_cfg['training']:
                        cfg_args['work_dir'] = pose_cfg['training']['work_dir']
                    if 'resume' in pose_cfg['training']:
                        cfg_args['resume'] = pose_cfg['training']['resume']
                    if 'load_from' in pose_cfg['training']:
                         cfg_args['load_from'] = pose_cfg['training']['load_from']
                
                if 'optimization' in pose_cfg and 'learning_rate' in pose_cfg['optimization']:
                     if 'base_lr' in pose_cfg['optimization']['learning_rate']:
                         cfg_args['lr'] = pose_cfg['optimization']['learning_rate']['base_lr']

    # Update defaults with config values
    final_args.update(cfg_args)
    
    # Override with command line args if they are not None
    cmd_args = {k: v for k, v in vars(args).items() if v is not None}
    final_args.update(cmd_args)
    
    # Check required
    if not final_args.get('data_root'):
         raise ValueError("data_root must be specified either in config or via command line")

    # Clean up non-training keys
    if 'config' in final_args:
        del final_args['config']
        
    print(f"Training with arguments: {final_args}")
    train(final_args)
