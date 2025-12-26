import torch
import cv2
import numpy as np
import argparse
import os
from models.rtmdet import RTMDet
from models.rtmpose import RTMPose
from utils.box_ops import distance2bbox, multiclass_nms

def preprocess_det(img, target_size=(640, 640)):
    # img: H, W, 3 (BGR)
    h, w, _ = img.shape
    img_resized = cv2.resize(img, target_size)
    tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
    return tensor.unsqueeze(0), (h, w) # Add batch dim

def postprocess_det(cls_scores, bbox_preds, input_shape, orig_shape, score_thr=0.3, nms_thr=0.5):
    # input_shape: (640, 640)
    # orig_shape: (H, W)
    all_bboxes = []
    all_scores = []
    
    device = cls_scores[0].device
    
    for level, (cls, reg) in enumerate(zip(cls_scores, bbox_preds)):
        # cls: (B, C, H, W)
        # reg: (B, 4, H, W)
        B, C, H, W = cls.shape
        stride = [8, 16, 32][level]
        
        cls = cls.permute(0, 2, 3, 1).reshape(B, -1, C).sigmoid()
        reg = reg.permute(0, 2, 3, 1).reshape(B, -1, 4) * stride
        
        # Grid
        shifts_x = torch.arange(0, W * stride, step=stride, dtype=torch.float32, device=device)
        shifts_y = torch.arange(0, H * stride, step=stride, dtype=torch.float32, device=device)
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing='ij')
        points = torch.stack([shift_x.reshape(-1), shift_y.reshape(-1)], dim=-1) # (N, 2)
        
        # Decode
        bboxes = distance2bbox(points[None].expand(B, -1, -1), reg)
        
        all_bboxes.append(bboxes)
        all_scores.append(cls)
        
    all_bboxes = torch.cat(all_bboxes, dim=1)[0]
    all_scores = torch.cat(all_scores, dim=1)[0]
    
    # NMS
    det_bboxes, det_scores, det_labels = multiclass_nms(all_bboxes, all_scores, score_thr, nms_thr)
    
    # Rescale to original image size
    # input was 640x640, orig was HxW
    scale_x = orig_shape[1] / input_shape[1]
    scale_y = orig_shape[0] / input_shape[0]
    
    if len(det_bboxes) > 0:
        det_bboxes[:, 0] *= scale_x
        det_bboxes[:, 1] *= scale_y
        det_bboxes[:, 2] *= scale_x
        det_bboxes[:, 3] *= scale_y
        
    return det_bboxes, det_scores, det_labels

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img', type=str, required=True, help='Path to image')
    parser.add_argument('--det_weights', type=str, default='train/weights/rtmdet_custom.pth')
    parser.add_argument('--pose_weights', type=str, default=None)
    parser.add_argument('--output', type=str, default='result.jpg')
    parser.add_argument('--score_thr', type=float, default=0.3)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Detection
    print(f"Loading RTMDet from {args.det_weights}...")
    det_model = RTMDet('s').to(device)
    det_model.load_state_dict(torch.load(args.det_weights, map_location=device))
    det_model.eval()
    
    img = cv2.imread(args.img)
    if img is None:
        raise FileNotFoundError(f"Image {args.img} not found")
        
    det_input, orig_shape = preprocess_det(img)
    det_input = det_input.to(device)
    
    input_h, input_w = 640, 640
    
    with torch.no_grad():
        cls_scores, bbox_preds = det_model(det_input)
        
    bboxes, scores, labels = postprocess_det(cls_scores, bbox_preds, (input_h, input_w), orig_shape, args.score_thr)
    
    print(f"Detected {len(bboxes)} objects.")
    
    # Visualization
    vis_img = img.copy()
    for bbox, score, label in zip(bboxes, scores, labels):
        x1, y1, x2, y2 = bbox.cpu().numpy().astype(int)
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(vis_img, f"{score:.2f}", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
    cv2.imwrite(args.output, vis_img)
    print(f"Saved result to {args.output}")

if __name__ == "__main__":
    main()
