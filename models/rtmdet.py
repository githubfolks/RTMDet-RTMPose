import torch
import torch.nn as nn
from .backbones.cspnext import CSPNeXt
from .necks.pafpn import CSPNeXtPAFPN
from .heads.rtmdet_head import RTMDetHead

class RTMDet(nn.Module):
    """
    RTMDet Object Detector.
    """
    def __init__(self, arch='s', num_classes=80):
        super().__init__()
        self.backbone = CSPNeXt(arch=arch)
        
        # Channels depend on arch. 
        # s: base=64, width=0.5 -> [64*0.5*2, 64*0.5*4, 64*0.5*8] = [64, 128, 256]?
        # Let's verify channel calc in CSPNeXt:
        # tiny: w=0.375. base=64.
        # s: w=0.5. base=64. stem=32.
        # Stages:
        # 1: out=32*2=64
        # 2: out=32*4=128
        # 3: out=32*8=256
        # 4: out=32*16=512
        # P3, P4, P5 corresponds to Stage 2, 3, 4 output.
        # So for 's', channels are [128, 256, 512].
        
        if arch == 's':
            in_channels = [128, 256, 512]
            neck_out_channels = 128
        elif arch == 'm':
            # w=0.75. base=64. stem=48.
            # 2: 48*4=192
            # 3: 48*8=384
            # 4: 48*16=768
            in_channels = [192, 384, 768]
            neck_out_channels = 256
        else:
            # Default fallback/placeholder
            in_channels = [128, 256, 512]
            neck_out_channels = 128
            
        self.neck = CSPNeXtPAFPN(in_channels=in_channels, out_channels=neck_out_channels)
        self.head = RTMDetHead(num_classes=num_classes, in_channels=neck_out_channels, feat_channels=neck_out_channels)
        
    def forward(self, x):
        # x: (B, 3, H, W)
        backbone_feats = self.backbone(x) # (P3, P4, P5) usually
        neck_feats = self.neck(backbone_feats)
        return self.head(neck_feats)

if __name__ == "__main__":
    model = RTMDet('s')
    x = torch.randn(1, 3, 640, 640)
    cls, box = model(x)
    print("RTMDet Output:", cls[0].shape)
