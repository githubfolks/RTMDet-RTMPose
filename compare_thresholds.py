"""
Generate visualizations with different confidence threshold configurations.
"""
import torch
import cv2
import numpy as np
import os
import glob
from models.rtmdet import RTMDet
from models.rtmpose import RTMPose
from infer import preprocess_det, postprocess_det

SKELETON_LINKS = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # Face
    (5, 6), (5, 7), (6, 8), (7, 9), (8, 10),  # Arms
    (5, 11), (6, 12), (11, 12),  # Torso
    (11, 13), (12, 14), (13, 15), (14, 16)  # Legs
]

# Different threshold configurations to test
THRESHOLD_CONFIGS = {
    'strict': {'hide': 0.5, 'high': 0.7, 'med': 0.6},      # Very strict
    'moderate': {'hide': 0.4, 'high': 0.65, 'med': 0.5},   # Moderate
    'default': {'hide': 0.3, 'high': 0.6, 'med': 0.4},     # Current default
    'lenient': {'hide': 0.2, 'high': 0.5, 'med': 0.35},    # More lenient
    'show_all': {'hide': 0.0, 'high': 0.6, 'med': 0.4},    # Show all keypoints
}

def decode_simcc(simcc_x, simcc_y, simcc_split_ratio=2.0):
    N, K, W_bins = simcc_x.shape
    locs_x = torch.argmax(simcc_x, dim=2).float()
    locs_y = torch.argmax(simcc_y, dim=2).float()
    conf_x = torch.max(torch.softmax(simcc_x, dim=2), dim=2)[0]
    conf_y = torch.max(torch.softmax(simcc_y, dim=2), dim=2)[0]
    confidence = (conf_x + conf_y) / 2
    locs_x /= simcc_split_ratio
    locs_y /= simcc_split_ratio
    return torch.stack([locs_x, locs_y], dim=-1), confidence

def get_color_by_confidence(conf, high_thresh, med_thresh):
    if conf >= high_thresh:
        return (0, 255, 0)  # Green
    elif conf >= med_thresh:
        return (0, 255, 255)  # Yellow
    else:
        return (0, 0, 255)  # Red

def process_image(img, det_model, pose_model, device, config, config_name):
    h_img, w_img, _ = img.shape
    hide_thresh = config['hide']
    high_thresh = config['high']
    med_thresh = config['med']
    
    det_input, orig_shape = preprocess_det(img)
    det_input = det_input.to(device)
    
    with torch.no_grad():
        cls_scores, bbox_preds = det_model(det_input)
    bboxes, scores, labels = postprocess_det(cls_scores, bbox_preds, (640, 640), orig_shape, 0.16, nms_thr=0.1)
    
    vis_img = img.copy()
    padding_ratio = 0.25
    
    valid_boxes = []
    for bbox, score in zip(bboxes, scores):
        x1, y1, x2, y2 = bbox.cpu().numpy().astype(int)
        if score < 0.3:
            if x1 < -2 or y1 < -2 or x2 > w_img + 2 or y2 > h_img + 2:
                continue
        w = x2 - x1
        h = y2 - y1
        center_x = x1 + w / 2 + w * 0.2
        new_w = w * 1.5
        x1_refined = int(center_x - new_w / 2)
        x2_refined = int(center_x + new_w / 2)
        padding_y = int(h * 0.1)
        y1_refined = y1 - padding_y
        y2_refined = y2 + padding_y
        valid_boxes.append((x1_refined, y1_refined, x2_refined, y2_refined, float(score)))
        cv2.rectangle(vis_img, (x1_refined, y1_refined), (x2_refined, y2_refined), (255, 0, 0), 2)
    
    total_keypoints = 0
    visible_keypoints = 0
    
    for person_idx, (x1_box, y1_box, x2_box, y2_box, det_score) in enumerate(valid_boxes):
        w_box = x2_box - x1_box
        h_box = y2_box - y1_box
        pad_w = int(w_box * padding_ratio)
        pad_h = int(h_box * padding_ratio)
        
        x1_crop = max(0, x1_box - pad_w)
        y1_crop = max(0, y1_box - pad_h)
        x2_crop = min(w_img, x2_box + pad_w)
        y2_crop = min(h_img, y2_box + pad_h)
        
        crop_w = x2_crop - x1_crop
        crop_h = y2_crop - y1_crop
        
        if crop_w < 10 or crop_h < 10:
            continue
        
        person_crop = img[y1_crop:y2_crop, x1_crop:x2_crop]
        pose_input_img = cv2.resize(person_crop, (192, 256))
        pose_input_img_rgb = cv2.cvtColor(pose_input_img, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(pose_input_img_rgb).permute(2, 0, 1).float() / 255.0
        tensor = tensor.unsqueeze(0).to(device)
        
        with torch.no_grad():
            simcc_x, simcc_y = pose_model(tensor)
        
        keypoints, confidence = decode_simcc(simcc_x, simcc_y)
        keypoints = keypoints[0].cpu().numpy()
        confidence = confidence[0].cpu().numpy()
        
        scale_x = crop_w / 192.0
        scale_y = crop_h / 256.0
        
        trans_keypoints = []
        for i, (pt, conf) in enumerate(zip(keypoints, confidence)):
            kx = int(pt[0] * scale_x) + x1_crop
            ky = int(pt[1] * scale_y) + y1_crop
            trans_keypoints.append((kx, ky, conf))
            total_keypoints += 1
            if conf >= hide_thresh:
                visible_keypoints += 1
        
        for i, (kx, ky, conf) in enumerate(trans_keypoints):
            if conf >= hide_thresh:
                color = get_color_by_confidence(conf, high_thresh, med_thresh)
                cv2.circle(vis_img, (kx, ky), 5, color, -1)
                cv2.circle(vis_img, (kx, ky), 5, (0, 0, 0), 1)
        
        for (u, v) in SKELETON_LINKS:
            if u < len(trans_keypoints) and v < len(trans_keypoints):
                pt1_x, pt1_y, conf1 = trans_keypoints[u]
                pt2_x, pt2_y, conf2 = trans_keypoints[v]
                if conf1 >= hide_thresh and conf2 >= hide_thresh:
                    min_conf = min(conf1, conf2)
                    color = get_color_by_confidence(min_conf, high_thresh, med_thresh)
                    cv2.line(vis_img, (pt1_x, pt1_y), (pt2_x, pt2_y), color, 2)
        
        cv2.putText(vis_img, f"P{person_idx+1}", (x1_box, y1_box - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    # Add config info
    cv2.putText(vis_img, f"Config: {config_name.upper()}", (10, 25), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(vis_img, f"Hide<{hide_thresh} | High>={high_thresh} | Med>={med_thresh}", 
               (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    cv2.putText(vis_img, f"Visible: {visible_keypoints}/{total_keypoints} keypoints", 
               (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    
    # Legend
    cv2.circle(vis_img, (w_img - 100, 25), 6, (0, 255, 0), -1)
    cv2.putText(vis_img, "High", (w_img - 85, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.circle(vis_img, (w_img - 100, 45), 6, (0, 255, 255), -1)
    cv2.putText(vis_img, "Med", (w_img - 85, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.circle(vis_img, (w_img - 100, 65), 6, (0, 0, 255), -1)
    cv2.putText(vis_img, "Low", (w_img - 85, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    return vis_img

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("Loading models...")
    det_model = RTMDet('s').to(device)
    det_model.load_state_dict(torch.load('train/weights/rtmdet_custom.pth', map_location=device))
    det_model.eval()
    
    pose_model = RTMPose('s', input_size=(256, 192)).to(device)
    pose_model.load_state_dict(torch.load('train/weights/rtmpose_custom.pth', map_location=device))
    pose_model.eval()
    
    # Process one test image with all configs
    test_image = 'sample_input/200000346.jpg'
    img = cv2.imread(test_image)
    
    os.makedirs('comparison/thresholds', exist_ok=True)
    
    print(f"\nProcessing {test_image} with different threshold configurations:")
    print("-" * 60)
    
    for config_name, config in THRESHOLD_CONFIGS.items():
        vis_img = process_image(img, det_model, pose_model, device, config, config_name)
        save_path = f'comparison/thresholds/{config_name}.jpg'
        cv2.imwrite(save_path, vis_img)
        print(f"  {config_name:12} -> Hide<{config['hide']:.1f}, High>={config['high']:.1f}, Med>={config['med']:.1f}")
    
    print(f"\nSaved to comparison/thresholds/")
    print("\nConfigurations:")
    print("  strict   - Only show very confident keypoints (hide < 0.5)")
    print("  moderate - Balanced approach (hide < 0.4)")
    print("  default  - Current settings (hide < 0.3)")
    print("  lenient  - Show more keypoints (hide < 0.2)")
    print("  show_all - Show all keypoints regardless of confidence")

if __name__ == "__main__":
    main()
