import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from models.rtmpose import RTMPose
from data.yolo import YOLODataset
import argparse
import torch.nn.functional as F
import datetime
import os

import cv2
import numpy as np

# Transform moved to global scope for multiprocessing
class ResizeTransform:
    def __init__(self, size=(192, 256)): # W, H
        self.size = size
        
    def __call__(self, img, target):
        # img: H, W, 3
        h, w, _ = img.shape
        target_w, target_h = self.size
        
        img_resized = cv2.resize(img, (target_w, target_h))
        
        # Scale keypoints
        scale_x = target_w / w
        scale_y = target_h / h

        if len(target['keypoints']) > 0:
            kpts = target['keypoints']
            kpts[:, :, 0] *= scale_x
            kpts[:, :, 1] *= scale_y
            target['keypoints'] = kpts
        return img_resized, target

def collate_fn(batch):
    imgs = []
    kpts_targets = []
    for img, target in batch:
        imgs.append(torch.from_numpy(img).permute(2, 0, 1).float() / 255.0)
        # Only taking the first person for "Single Person Pose" simplicity?
        # Or Top-Down usually crops per person.
        # If our dataset has full images with keypoints, we are training a "Bottom-up" or "Top-down" approach?
        # RTMPose is Top-Down. It expects a cropped person.
        # If the user's dataset is full images with one person, it works.
        # If multiple people, we need to crop them out.
        # Assuming Single Person or Primary Person for this basic script
        
        kpts = target['keypoints']
        # Take first person with valid keypoints
        if len(kpts) > 0:
             kpts_targets.append(kpts[0]) # (K, 3)
        else:
             # Dummy for empty?
             kpts_targets.append(torch.zeros(17, 3))
             
    return torch.stack(imgs), torch.stack(kpts_targets)

def train(item_dict):
    data_root = item_dict['data_root']
    epochs = item_dict['epochs']
    batch_size = item_dict['batch_size']
    resume_path = item_dict.get('resume', None)
    work_dir = item_dict.get('work_dir', 'train/weights')
    
    os.makedirs(work_dir, exist_ok=True)

    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Training on device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Dataset
    # Split='train' handles getting images/labels from train folders
    dataset = YOLODataset(data_root, split='train', use_pose=True)
    
    # Collate: handle variable size or just stack if resize is done in dataset?
    # Our YOLODataset currently doesn't resize images to fixed size!
    # The Model expects fixed size (e.g. 256x192).
    # We NEED a transform to resize image and keypoints!
    
    # For now, let's add a simple resize transform inline or in dataset
    # But for robustness, let's rely on a proper transform.
    # To keep this "Scratch" simple, I'll modify collate to resize or assume user pre-resized?
    # No, model needs fixed input.
    # I will add a resize logic to the Dataset `__getitem__` if transform is None.
    
    # Actually, let's define a simple transform here
    import cv2
    import numpy as np
    
    # Transform defined globally now

    # Pass transform to dataset? Dataset supports it.
    # Re-instantiate with transform
    dataset.transform = ResizeTransform(size=(192, 256))
    
    # Collate defined globally now
        
    # When using RAM Cache on Windows, num_workers must be 0 to avoid expensive pickling of the huge dataset object
    num_workers = 0 if dataset.cache_ram else 4
    persistent_workers = False if num_workers == 0 else True

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn,
                            num_workers=num_workers, pin_memory=True, persistent_workers=persistent_workers)
    
    # Model
    model = RTMPose('s', input_size=(256, 192)).to(device)
    
    if resume_path and os.path.exists(resume_path):
        print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Resuming from {resume_path}...")
        try:
            model.load_state_dict(torch.load(resume_path, map_location=device))
        except Exception as e:
            print(f"Warning: Failed to resume: {e}")
            
    model.train()
    
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.05)
    
    # SimCC Loss Logic (KL Div)
    # We need to turn GT keypoints into SimCC targets (1D Vectors)
    def simcc_loss(pred_simcc_x, pred_simcc_y, gt_kpts, simcc_split_ratio=2.0, sigma=6.0):
        # gt_kpts: (B, K, 3)
        B, K, W_bins = pred_simcc_x.shape
        _, _, H_bins = pred_simcc_y.shape
        
        # Generate generic 1D Gaussian targets
        # ... Implementation of label smoothing/Gaussian generation on 1D ...
        
        # Simplified: Cross Entropy with "One Hot" or "Soft Label" of coordinate index
        # x_loc = gt_x * split_ratio
        
        loss = 0
        valid_mask = gt_kpts[..., 2] > 0
        
        if valid_mask.sum() == 0:
             return torch.tensor(0.0, device=device, requires_grad=True)
             
        # Create targets
        gt_x = gt_kpts[..., 0] * simcc_split_ratio
        gt_y = gt_kpts[..., 1] * simcc_split_ratio
        
        # Clip to bins
        gt_x = gt_x.clamp(0, W_bins-1).long()
        gt_y = gt_y.clamp(0, H_bins-1).long()
        
        # Simple Cross Entropy for classification
        # Flatten
        loss_x = F.cross_entropy(pred_simcc_x.permute(0, 2, 1), gt_x, reduction='none')
        loss_y = F.cross_entropy(pred_simcc_y.permute(0, 2, 1), gt_y, reduction='none')
        
        loss = (loss_x * valid_mask + loss_y * valid_mask).sum() / (valid_mask.sum() + 1e-6)
        return loss

    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Starting Pose Training...")
    for epoch in range(epochs):
        total_loss = 0
        for i, (imgs, gt_kpts) in enumerate(dataloader):
            imgs = imgs.to(device)
            gt_kpts = gt_kpts.to(device)
            
            optimizer.zero_grad()
            
            # Forward
            pred_x, pred_y = model(imgs)
            
            loss = simcc_loss(pred_x, pred_y, gt_kpts)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if i % 10 == 0:
                print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Epoch [{epoch+1}/{epochs}], Step [{i}/{len(dataloader)}], Loss: {loss.item():.4f}")
                
        print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Epoch {epoch+1} Complete. Avg Loss: {total_loss/len(dataloader):.4f}")
        
        # Save every epoch
        save_path = os.path.join(work_dir, 'rtmpose_custom.pth')
        torch.save(model.state_dict(), save_path)
        
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Pose Training Complete. Model saved to {save_path}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=50) # Increased to 50
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--work_dir', type=str, default='train/weights', help='Directory to save weights')
    args = parser.parse_args()
    
    train(vars(args))
