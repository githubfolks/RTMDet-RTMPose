"""
Visualize sample annotations to verify data quality.
"""
import cv2
import numpy as np
import os
import random

KEYPOINT_NAMES = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]

SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # Face
    (5, 6), (5, 7), (6, 8), (7, 9), (8, 10),  # Arms
    (5, 11), (6, 12), (11, 12),  # Torso
    (11, 13), (12, 14), (13, 15), (14, 16)  # Legs
]

def parse_label_file(filepath):
    annotations = []
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id = int(parts[0])
            cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            keypoints = []
            kpt_data = parts[5:]
            for i in range(0, len(kpt_data), 3):
                if i + 2 < len(kpt_data):
                    kx = float(kpt_data[i])
                    ky = float(kpt_data[i + 1])
                    vis = int(float(kpt_data[i + 2]))
                    keypoints.append((kx, ky, vis))
            annotations.append({
                'class': cls_id,
                'bbox': (cx, cy, w, h),
                'keypoints': keypoints
            })
    return annotations

def visualize_annotation(img_path, label_path, output_path):
    img = cv2.imread(img_path)
    if img is None:
        return
    h, w = img.shape[:2]
    
    annotations = parse_label_file(label_path)
    
    for ann in annotations:
        cx, cy, bw, bh = ann['bbox']
        
        # Convert normalized to pixel coordinates
        x1 = int((cx - bw/2) * w)
        y1 = int((cy - bh/2) * h)
        x2 = int((cx + bw/2) * w)
        y2 = int((cy + bh/2) * h)
        
        # Draw bbox
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        # Draw keypoints
        kpts_px = []
        for kx, ky, vis in ann['keypoints']:
            px = int(kx * w)
            py = int(ky * h)
            kpts_px.append((px, py, vis))
            
            if vis == 2:
                cv2.circle(img, (px, py), 4, (0, 255, 0), -1)  # Green - visible
            elif vis == 1:
                cv2.circle(img, (px, py), 4, (0, 255, 255), -1)  # Yellow - occluded
            else:
                cv2.circle(img, (px, py), 4, (0, 0, 255), -1)  # Red - not labeled
        
        # Draw skeleton
        for (u, v) in SKELETON:
            if u < len(kpts_px) and v < len(kpts_px):
                pt1 = kpts_px[u]
                pt2 = kpts_px[v]
                if pt1[2] >= 1 and pt2[2] >= 1:
                    cv2.line(img, (pt1[0], pt1[1]), (pt2[0], pt2[1]), (0, 255, 0), 1)
    
    cv2.imwrite(output_path, img)

def main():
    img_dir = 'custom_dataset/images/train'
    label_dir = 'custom_dataset/labels/train'
    output_dir = 'comparison/ground_truth'
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Get sample images (random + test images if they exist in train)
    all_images = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]
    
    # Prioritize test images that are also in sample_input
    test_images = ['200000274.jpg', '200000278.jpg', '200000308.jpg', '200000346.jpg']
    sample_images = [f for f in test_images if f in all_images]
    
    # Add some random samples
    random.seed(42)
    random_samples = random.sample(all_images, min(6, len(all_images)))
    sample_images.extend([f for f in random_samples if f not in sample_images])
    
    print(f"Visualizing {len(sample_images)} sample annotations...")
    
    for img_name in sample_images[:10]:
        img_path = os.path.join(img_dir, img_name)
        label_path = os.path.join(label_dir, img_name.replace('.jpg', '.txt'))
        output_path = os.path.join(output_dir, f'gt_{img_name}')
        
        if os.path.exists(img_path) and os.path.exists(label_path):
            visualize_annotation(img_path, label_path, output_path)
            print(f"  Saved: {output_path}")
        else:
            print(f"  Skipped: {img_name} (files not found)")
    
    print(f"\nGround truth visualizations saved to {output_dir}/")

if __name__ == "__main__":
    main()
