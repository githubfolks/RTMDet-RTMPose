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

    
    # Device selection: CUDA > MPS (Mac) > CPU
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Training on device: {device}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Training on device: {device} (Apple Silicon GPU)")
    else:
        device = torch.device('cpu')
        print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Training on device: {device}")
    
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
    dataset.transform = CropPersonTransform(size=(192, 256), padding_ratio=0.25)
    
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
