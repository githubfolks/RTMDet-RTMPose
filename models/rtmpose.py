import torch
import torch.nn as nn
from .backbones.cspnext import CSPNeXt
from .heads.rtmpose_head import SimCCHead

class RTMPose(nn.Module):
    """
    RTMPose Pose Estimator.
    Usually uses a backbone (P5) -> Head.
    """
    def __init__(self, arch='s', num_keypoints=17, input_size=(256, 192)):
        super().__init__()
        # RTMPose often uses 'CSPNeXt' but optimized for pose (sometimes loading pretrained detection backbone).
        self.backbone = CSPNeXt(arch=arch, out_indices=(4,)) # Only need P5 usually
        
        if arch == 's':
            in_channels = 512
        elif arch == 'm':
            in_channels = 768
        else:
            in_channels = 512
            
        # Calculate feature map size for P5 (stride 32)
        feat_h = input_size[0] // 32
        feat_w = input_size[1] // 32
        
        self.head = SimCCHead(
            in_channels=in_channels, 
            out_channels=num_keypoints, 
            input_size=input_size, 
            in_feature_size=(feat_h, feat_w)
        )
        
    def forward(self, x):
        # x: (B, 3, H, W)
        feats = self.backbone(x)
        # feats is tuple, we want last one
        return self.head(feats[-1])

if __name__ == "__main__":
    model = RTMPose('s')
    x = torch.randn(1, 3, 256, 192)
    px, py = model(x)
    print("RTMPose Output:", px.shape, py.shape)
