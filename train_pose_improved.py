"""
Improved RTMPose Training Script
With better augmentation, learning rate scheduling, and handling of edge cases.
"""
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from models.rtmpose import RTMPose
from data.yolo import YOLODataset
import argparse
import torch.nn.functional as F
import datetime
import os
import random
import cv2
import numpy as np
import math

# ============================================
# DATA AUGMENTATION
# ============================================

class ImprovedCropPersonTransform:
    """
    Enhanced Top-Down transform with augmentation.
    - Random scaling
    - Random rotation
    - Color jittering
    - Random horizontal flip
    """
    def __init__(self, size=(192, 256), padding_ratio=0.25, 
                 augment=True, 
                 scale_range=(0.75, 1.25),
                 rotation_range=(-30, 30),
                 flip_prob=0.5,
                 color_jitter=True):
        self.size = size  # W, H
        self.padding_ratio = padding_ratio
        self.augment = augment
        self.scale_range = scale_range
        self.rotation_range = rotation_range
        self.flip_prob = flip_prob
        self.color_jitter = color_jitter
        
        # Keypoint flip pairs (COCO format)
        self.flip_pairs = [
            (1, 2),   # eyes
            (3, 4),   # ears
            (5, 6),   # shoulders
            (7, 8),   # elbows
            (9, 10),  # wrists
            (11, 12), # hips
            (13, 14), # knees
            (15, 16), # ankles
        ]
        
    def __call__(self, img, target):
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
            
            if w <= 0 or h <= 0:
                continue
            
            # Random scale augmentation
            if self.augment:
                scale = random.uniform(*self.scale_range)
            else:
                scale = 1.0
            
            # Calculate padding with scale
            pad_w = w * self.padding_ratio * scale
            pad_h = h * self.padding_ratio * scale
            
            # Expand bbox by scale factor
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            new_w = w * scale
            new_h = h * scale
            
            x1_scaled = center_x - new_w / 2
            y1_scaled = center_y - new_h / 2
            x2_scaled = center_x + new_w / 2
            y2_scaled = center_y + new_h / 2
            
            x1_pad = max(0, int(x1_scaled - pad_w))
            y1_pad = max(0, int(y1_scaled - pad_h))
            x2_pad = min(w_img, int(x2_scaled + pad_w))
            y2_pad = min(h_img, int(y2_scaled + pad_h))
            
            crop = img[y1_pad:y2_pad, x1_pad:x2_pad].copy()
            crop_h, crop_w = crop.shape[:2]
            
            if crop_w < 10 or crop_h < 10:
                continue
            
            # ---------- AUGMENTATIONS ----------
            
            # Color jittering
            if self.augment and self.color_jitter:
                crop = self.apply_color_jitter(crop)
            
            # Horizontal flip
            do_flip = self.augment and random.random() < self.flip_prob
            if do_flip:
                crop = cv2.flip(crop, 1)
            
            # Rotation (applied after resize for efficiency)
            if self.augment and self.rotation_range[1] > 0:
                angle = random.uniform(*self.rotation_range)
            else:
                angle = 0
            
            # Resize crop to target size
            crop_resized = cv2.resize(crop, (target_w, target_h))
            
            # Apply rotation if needed
            if angle != 0:
                crop_resized, rot_matrix = self.rotate_image(crop_resized, angle)
            else:
                rot_matrix = None
            
            # Transform keypoints
            scale_x = target_w / crop_w
            scale_y = target_h / crop_h
            
            transformed_kpts = []
            for j, kpt in enumerate(kpts):
                kx, ky, kv = kpt
                if kv > 0:
                    # Shift to crop origin, then scale
                    new_kx = (kx - x1_pad) * scale_x
                    new_ky = (ky - y1_pad) * scale_y
                    
                    # Apply flip
                    if do_flip:
                        new_kx = target_w - 1 - new_kx
                    
                    # Apply rotation
                    if rot_matrix is not None:
                        new_kx, new_ky = self.rotate_point(new_kx, new_ky, rot_matrix, target_w, target_h)
                    
                    # Clamp to valid range
                    new_kx = max(0, min(target_w - 1, new_kx))
                    new_ky = max(0, min(target_h - 1, new_ky))
                else:
                    new_kx, new_ky = 0, 0
                    
                transformed_kpts.append([new_kx, new_ky, kv])
            
            # Swap keypoints for flip
            if do_flip:
                transformed_kpts = self.flip_keypoints(transformed_kpts)
            
            cropped_samples.append((crop_resized, np.array(transformed_kpts)))
        
        return cropped_samples
    
    def apply_color_jitter(self, img):
        """Apply random color jittering."""
        # Brightness
        brightness = random.uniform(0.7, 1.3)
        img = np.clip(img * brightness, 0, 255).astype(np.uint8)
        
        # Contrast
        contrast = random.uniform(0.7, 1.3)
        mean = np.mean(img)
        img = np.clip((img - mean) * contrast + mean, 0, 255).astype(np.uint8)
        
        # Saturation (convert to HSV)
        if random.random() < 0.5:
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:, :, 1] *= random.uniform(0.7, 1.3)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
            img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        
        return img
    
    def rotate_image(self, img, angle):
        """Rotate image by angle degrees."""
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(img, rot_matrix, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(128, 128, 128))
        return rotated, rot_matrix
    
    def rotate_point(self, x, y, rot_matrix, w, h):
        """Rotate a point using the rotation matrix."""
        point = np.array([x, y, 1])
        rotated = rot_matrix @ point
        return rotated[0], rotated[1]
    
    def flip_keypoints(self, kpts):
        """Swap left-right keypoints after horizontal flip."""
        kpts = list(kpts)
        for left, right in self.flip_pairs:
            kpts[left], kpts[right] = kpts[right], kpts[left]
        return kpts


def collate_fn(batch):
    """Collate function for Top-Down pose training."""
    imgs = []
    kpts_targets = []
    
    for samples in batch:
        for crop, kpts in samples:
            imgs.append(torch.from_numpy(crop).permute(2, 0, 1).float() / 255.0)
            kpts_targets.append(torch.tensor(kpts, dtype=torch.float32))
    
    if len(imgs) == 0:
        return torch.zeros(1, 3, 256, 192), torch.zeros(1, 17, 3)
    
    return torch.stack(imgs), torch.stack(kpts_targets)


# ============================================
# LOSS FUNCTION
# ============================================

def simcc_loss_improved(pred_simcc_x, pred_simcc_y, gt_kpts, simcc_split_ratio=2.0, sigma=6.0, use_soft_label=True):
    """
    Improved SimCC loss with optional soft labels (Gaussian smoothing).
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


# ============================================
# TRAINING FUNCTION
# ============================================

def train(item_dict):
    data_root = item_dict['data_root']
    epochs = item_dict['epochs']
    batch_size = item_dict['batch_size']
    resume_path = item_dict.get('resume', None)
    work_dir = item_dict.get('work_dir', 'train/weights')
    base_lr = item_dict.get('lr', 0.001)
    use_augmentation = item_dict.get('augment', True)
    use_soft_label = item_dict.get('soft_label', True)
    
    os.makedirs(work_dir, exist_ok=True)
    
    # Device selection
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Training on: {device} ({torch.cuda.get_device_name(0)})")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Training on: {device} (Apple Silicon GPU)")
    else:
        device = torch.device('cpu')
        print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Training on: {device}")
    
    # Dataset with improved transform
    dataset = YOLODataset(data_root, split='train', use_pose=True)
    
    transform = ImprovedCropPersonTransform(
        size=(192, 256), 
        padding_ratio=0.25,
        augment=use_augmentation,
        scale_range=(0.75, 1.25),
        rotation_range=(-30, 30),
        flip_prob=0.5,
        color_jitter=True
    )
    dataset.transform = transform
    
    num_workers = 0 if dataset.cache_ram else 4
    persistent_workers = False if num_workers == 0 else True
    
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=num_workers, 
        pin_memory=True, 
        persistent_workers=persistent_workers
    )
    
    # Model
    model = RTMPose('s', input_size=(256, 192)).to(device)
    
    if resume_path and os.path.exists(resume_path):
        print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Resuming from {resume_path}...")
        try:
            model.load_state_dict(torch.load(resume_path, map_location=device))
        except Exception as e:
            print(f"Warning: Failed to resume: {e}")
    
    model.train()
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=0.05)
    
    # Learning rate schedulers
    # Warmup for 5 epochs, then cosine annealing
    warmup_epochs = 5
    warmup_scheduler = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs, eta_min=base_lr * 0.01)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs])
    
    # Print config
    print(f"\n{'='*60}")
    print("IMPROVED TRAINING CONFIGURATION")
    print(f"{'='*60}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Base LR: {base_lr}")
    print(f"  Warmup epochs: {warmup_epochs}")
    print(f"  Augmentation: {use_augmentation}")
    print(f"  Soft labels: {use_soft_label}")
    print(f"  Data samples: {len(dataset)}")
    print(f"{'='*60}\n")
    
    best_loss = float('inf')
    
    for epoch in range(epochs):
        total_loss = 0
        current_lr = optimizer.param_groups[0]['lr']
        
        for i, (imgs, gt_kpts) in enumerate(dataloader):
            imgs = imgs.to(device)
            gt_kpts = gt_kpts.to(device)
            
            optimizer.zero_grad()
            
            pred_x, pred_y = model(imgs)
            loss = simcc_loss_improved(pred_x, pred_y, gt_kpts, use_soft_label=use_soft_label)
            
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            
            if i % 20 == 0:
                print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Epoch [{epoch+1}/{epochs}], "
                      f"Step [{i}/{len(dataloader)}], Loss: {loss.item():.4f}, LR: {current_lr:.6f}")
        
        scheduler.step()
        
        avg_loss = total_loss / len(dataloader)
        print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Epoch {epoch+1} Complete. "
              f"Avg Loss: {avg_loss:.4f}, LR: {current_lr:.6f}")
        
        # Save checkpoint
        save_path = os.path.join(work_dir, 'rtmpose_custom.pth')
        torch.save(model.state_dict(), save_path)
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = os.path.join(work_dir, 'rtmpose_best.pth')
            torch.save(model.state_dict(), best_path)
            print(f"  -> New best model saved (loss: {best_loss:.4f})")
    
    print(f"\n[{datetime.datetime.now().strftime('%H:%M:%S')}] Training Complete!")
    print(f"  Final model: {save_path}")
    print(f"  Best model: {os.path.join(work_dir, 'rtmpose_best.pth')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Improved RTMPose Training")
    parser.add_argument('--data_root', type=str, required=True, help='Path to dataset')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Base learning rate')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint')
    parser.add_argument('--work_dir', type=str, default='train/weights', help='Output directory')
    parser.add_argument('--no_augment', action='store_true', help='Disable augmentation')
    parser.add_argument('--no_soft_label', action='store_true', help='Use hard labels instead of soft')
    args = parser.parse_args()
    
    config = {
        'data_root': args.data_root,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'lr': args.lr,
        'resume': args.resume,
        'work_dir': args.work_dir,
        'augment': not args.no_augment,
        'soft_label': not args.no_soft_label,
    }
    
    train(config)
