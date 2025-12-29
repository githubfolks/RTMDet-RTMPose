import torch
import torch.nn as nn
from ..layers.blocks import ConvModule

class RTMDetHead(nn.Module):
    """
    RTMDet Head.
    
    Args:
        num_classes (int): Number of classes.
        in_channels (int): Input channels.
        feat_channels (int): Hidden channels in head.
        stacked_convs (int): Number of convs in head.
        strides (list): Strides of each level.
    """
    def __init__(self, num_classes=80, in_channels=128, feat_channels=128, stacked_convs=2, strides=[8, 16, 32]):
        super().__init__()
        self.num_classes = num_classes
        self.strides = strides
        
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        
        # Shared weights across levels, but separate BN stats
        # To likely simplify, we'll just create identical modules for each stride for now
        # OR implementation of shared-weight-conv-module if strict RTMDet adherence is needed.
        # For "Scratch" implementation, independent heads per level is easier but heavier.
        # RTMDet uses "shared convs, separate BN".
        
        # Let's implement the Shared-Conv-Sep-BN logic if possible, 
        # but for simplicity and robustness in this first pass, we can just use independent heads.
        # However, to be "RTMDet", we should try to share.
        
        # Simplified: Independent heads per level (Slight parameter overhead, same logic)
        for _ in strides:
            cls_conv = []
            reg_conv = []
            for _ in range(stacked_convs):
                cls_conv.append(ConvModule(in_channels, feat_channels, 3, 1, 1))
                reg_conv.append(ConvModule(in_channels, feat_channels, 3, 1, 1))
            

            self.cls_convs.append(nn.Sequential(*cls_conv, nn.Conv2d(feat_channels, num_classes, 3, 1, 1)))
            self.reg_convs.append(nn.Sequential(*reg_conv, nn.Conv2d(feat_channels, 4, 3, 1, 1)))
            
        self.init_weights()
            
    def init_weights(self):
        # Initialize CLS branch with bias -4.59 (prob ~0.01)
        for m in self.cls_convs:
            if isinstance(m, nn.Sequential):
                # Last layer is conv
                nn.init.constant_(m[-1].bias, -4.59)
                
    def forward(self, feats):
        cls_scores = []
        bbox_preds = []
        
        for i, x in enumerate(feats):
            cls_out = self.cls_convs[i](x)
            reg_out = self.reg_convs[i](x)
            

            # Use softplus instead of exp for numerical stability
            # softplus(x) = log(1 + exp(x)), which is bounded and more stable
            # Clamp input first to prevent overflow
            reg_out = reg_out.clamp(min=-20, max=20)
            reg_out = torch.nn.functional.softplus(reg_out, beta=1, threshold=20)
            
            cls_scores.append(cls_out)
            bbox_preds.append(reg_out)
            
        return cls_scores, bbox_preds

    def loss(self, cls_scores, bbox_preds, gt_bboxes, gt_labels, img_metas=None):
        """
        Compute loss.
        """
        from utils.losses import SimOTAAssigner, quality_focal_loss, giou_loss
        from utils.box_ops import distance2bbox
        
        assigner = SimOTAAssigner()
        device = cls_scores[0].device
        
        # 1. Flatten all levels
        all_cls_scores = []
        all_bbox_preds = []
        all_anchors = []
        
        # Generate anchors on the fly for all levels
        for i, stride in enumerate(self.strides):
             cls_feat = cls_scores[i] # (B, C, H, W)
             reg_feat = bbox_preds[i] # (B, 4, H, W) -> Already EXP'd in forward
             
             B, C, H, W = cls_feat.shape
             
             # Permute to (B, H*W, C)
             cls_feat = cls_feat.permute(0, 2, 3, 1).reshape(B, -1, C)
             reg_feat = reg_feat.permute(0, 2, 3, 1).reshape(B, -1, 4)
             
             all_cls_scores.append(cls_feat)
             all_bbox_preds.append(reg_feat)
             
             # Anchors (Centers)
             y, x = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
             # Shift +0.5 for center
             cx = (x.reshape(-1) + 0.5) * stride
             cy = (y.reshape(-1) + 0.5) * stride
             anchors = torch.stack([cx, cy, torch.full_like(cx, stride)], dim=-1) # (N, 3)
             all_anchors.append(anchors.expand(B, -1, -1))

        all_cls_scores = torch.cat(all_cls_scores, dim=1) # (B, N_total, C)
        all_bbox_preds = torch.cat(all_bbox_preds, dim=1) # (B, N_total, 4)
        all_anchors = torch.cat(all_anchors, dim=1)       # (B, N_total, 3)
        
        total_loss_cls = 0
        total_loss_bbox = 0
        num_pos_total = 0
        
        # Loop over batch (Simplification: Sequential assignment per image)
        for b_idx in range(B):
            pred_cls = all_cls_scores[b_idx] # (N, C)
            pred_reg = all_bbox_preds[b_idx] # (N, 4)
            anchors_img = all_anchors[b_idx]  # (N, 3)
            
            # Decode Boxes for Assignment
            stride_tensor = anchors_img[:, 2][:, None]
            pred_boxes_decoded = distance2bbox(anchors_img[:, :2], pred_reg * stride_tensor)
            
            # GT for this image
            # Assuming gt_bboxes is list of tensors or padded
            gts = gt_bboxes[b_idx]
            lbls = gt_labels[b_idx]
            
            # Assign
            assigned_gt_inds = assigner.assign(pred_cls, pred_boxes_decoded, anchors_img, gts, lbls)
            
            pos_mask = assigned_gt_inds >= 0
            num_pos = pos_mask.sum()
            num_pos_total += num_pos
            
            # CLS Loss
            # Target construction: (N, C)
            target_cls = torch.zeros_like(pred_cls)
            if num_pos > 0:
                pos_inds = torch.where(pos_mask)[0]
                pos_gt_inds = assigned_gt_inds[pos_inds]
                
                pos_gts = gts[pos_gt_inds]
                pos_lbls = lbls[pos_gt_inds]
                
                # QFL Target: IoU between pred and gt
                pos_pred_boxes = pred_boxes_decoded[pos_mask]
                
                # Calculate IoU for quality score (0-1)
                # Need IoU function
                # ... Simplified: Just using 1.0 for now as hard target is easier first step
                # RTMDet uses IoU Score.
                
                # Set targets
                # Using 1.0 for target category
                target_cls[pos_inds, pos_lbls] = 1.0
                
                # BBox Loss (Only positives)
                loss_bbox = giou_loss(pos_pred_boxes, pos_gts)
                total_loss_bbox += loss_bbox * 2.0 # Weight
                
            loss_cls = quality_focal_loss(pred_cls, target_cls)
            total_loss_cls += loss_cls

        if num_pos_total > 0:
            total_loss_cls /= num_pos_total
            total_loss_bbox /= B # Already mean inside giou? usually per image norm
        else:
            # No positive samples in batch - return small placeholder loss
            total_loss_cls = torch.tensor(0.1, device=device, requires_grad=True)
            total_loss_bbox = torch.tensor(0.1, device=device, requires_grad=True)
        
        return {"loss_cls": total_loss_cls, "loss_bbox": total_loss_bbox}

if __name__ == "__main__":
    head = RTMDetHead()
    feats = [torch.randn(1, 128, 80, 80), torch.randn(1, 128, 40, 40), torch.randn(1, 128, 20, 20)]
    c, b = head(feats)
    print([x.shape for x in c])
