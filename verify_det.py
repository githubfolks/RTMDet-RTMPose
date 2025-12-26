import torch
import cv2
import numpy as np
from models.rtmdet import RTMDet
from utils.box_ops import distance2bbox, multiclass_nms
import os

def run_verify(image_path, weight_path, output_path='sample_output/result.jpg'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running inference on {device}...")
    
    # 1. Load Model
    # Important: Model must match training config (s size)
    model = RTMDet('s').to(device)
    
    # Load weights
    if os.path.exists(weight_path):
        state_dict = torch.load(weight_path, map_location=device)
        model.load_state_dict(state_dict)
        print("Weights loaded successfully.")
    else:
        print(f"Error: Weights not found at {weight_path}")
        return

    model.eval()
    
    # 2. Preprocess Image
    img_raw = cv2.imread(image_path)
    if img_raw is None:
        print(f"Error: Could not read image {image_path}")
        return
        
    h, w, c = img_raw.shape
    input_size = (640, 640)
    
    img_resized = cv2.resize(img_raw, input_size)
    img_tensor = torch.from_numpy(img_resized).float() / 255.0
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).to(device) # (1, 3, 640, 640)
    
    # 3. Inference
    with torch.no_grad():
        cls_scores, bbox_preds = model(img_tensor)
        
    # 4. Post-process (NMS)
    # RTMDet Head outputs list of features. We need to decode and NMS.
    # The head usually does this in `predict` or we do it manually.
    # For transparency, let's do it manually similar to reference code.
    
    decoded_bboxes = []
    decoded_scores = []
    decoded_labels = []
    
    strides = model.head.strides
    
    for i, stride in enumerate(strides):
        cls_score = cls_scores[i] # (1, C, H, W)
        bbox_pred = bbox_preds[i] # (1, 4, H, W)
        
        B, C, H, W = cls_score.shape
        
        # Center points
        y, x = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
        cx = (x.reshape(-1) + 0.5) * stride
        cy = (y.reshape(-1) + 0.5) * stride
        centers = torch.stack([cx, cy], dim=-1) # (N, 2)
        
        # Reshape preds
        cls_score = cls_score.permute(0, 2, 3, 1).reshape(-1, C).sigmoid()
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
        
        # Filter low confidence
        max_scores, _ = cls_score.max(dim=1)
        valid_mask = max_scores > 0.05
        
        if valid_mask.sum() == 0:
            continue
            
        cls_score = cls_score[valid_mask]
        bbox_pred = bbox_pred[valid_mask]
        centers = centers[valid_mask]
        
        # Decode boxes
        bboxes = distance2bbox(centers, bbox_pred) # (N_valid, 4) xyxy
        
        decoded_bboxes.append(bboxes)
        decoded_scores.append(cls_score)
        
    if len(decoded_bboxes) == 0:
        print("No detections found.")
        return

    all_bboxes = torch.cat(decoded_bboxes, dim=0)
    all_scores = torch.cat(decoded_scores, dim=0)
    
    # NMS
    # multiclass_nms(multi_bboxes, multi_scores, score_thr, nms_thr, max_num=100)
    dets, scores, labels = multiclass_nms(all_bboxes, all_scores, score_thr=0.35, nms_thr=0.45)
    
    print(f"Found {len(dets)} detections.")
    print(f"Top 10 Scores: {scores[:10]}")
    
    # 5. Visualize
    # Scale boxes back to original image size
    scale_x = w / input_size[0]
    scale_y = h / input_size[1]
    
    dets[:, 0] *= scale_x
    dets[:, 1] *= scale_y
    dets[:, 2] *= scale_x
    dets[:, 3] *= scale_y
    
    for i, box in enumerate(dets):
        x1, y1, x2, y2 = box.cpu().numpy().astype(int)
        score = scores[i].item()
        
        cv2.rectangle(img_raw, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img_raw, f"{score:.2f}", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
    cv2.imwrite(output_path, img_raw)
    print(f"Saved result to {output_path}")

if __name__ == "__main__":
    # Test on one of the validation images listed previously
    img_path = "sample_images/900000027.jpg" 
    filename = os.path.basename(img_path)
    output_path = os.path.join('sample_output', filename)
    run_verify(img_path, "train/weight/rtmdet_custom.pth", output_path=output_path)
