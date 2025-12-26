import torch
import torch.nn as nn
from ..layers.blocks import ConvModule, CSPLayer

class CSPNeXt(nn.Module):
    """
    CSPNeXt Backbone for RTMDet/RTMPose.
    
    Args:
        arch (str): 'tiny', 's', 'm', 'l', 'x'
        in_channels (int): Input channels (default 3)
        out_indices (tuple): Output stages (usually (2, 3, 4))
    """
    # [depth, width]
    arch_settings = {
        'tiny': [0.33, 0.375],
        's': [0.33, 0.50],
        'm': [0.67, 0.75],
        'l': [1.00, 1.00],
        'x': [1.33, 1.25]
    }
    
    def __init__(self, arch='s', in_channels=3, out_indices=(2, 3, 4)):
        super().__init__()
        self.out_indices = out_indices
        depth_mul, width_mul = self.arch_settings[arch]
        
        # Base channels configuration
        base_channels = 64
        base_depth = 3
        
        # Stem
        stem_channels = int(base_channels * width_mul)
        self.stem = nn.Sequential(
            ConvModule(in_channels, stem_channels // 2, 3, 2, 1),
            ConvModule(stem_channels // 2, stem_channels // 2, 3, 1, 1),
            ConvModule(stem_channels // 2, stem_channels, 3, 1, 1)
        )
        
        # Stages
        self.stages = nn.ModuleList()
        
        # Stage configuration: [in_channel_mult, out_channel_mult, num_blocks, stride]
        # Stage 1: stride 2
        # Stage 2: stride 2
        # Stage 3: stride 2
        # Stage 4: stride 2
        stage_cfg = [
            [1, 2, 3, 2],    # Stage 1 (P2/4)
            [2, 4, 6, 2],    # Stage 2 (P3/8)
            [4, 8, 6, 2],    # Stage 3 (P4/16)
            [8, 16, 3, 2]    # Stage 4 (P5/32) -- Wait, if weights imply 256, this should be 8? 
            # Actually, standard CSPNext usually doubles. 
            # BUT RTMDet-s weights demand P5=256. 
            # Let's trust the weights.
            # Change multiplier 16 -> 8 ?? 
            # Or is it [96, 192, 384]? 384 is 1.5x?
            # 512 -> 768 mismatch.
            # 256 -> 512.
            # If I set this to [8, 8, 3, 2], output is 256.
            # Let's try [8, 16, 3, 2] -> [256, 512].
            # Error was 256 vs 384?

        ]
        
        in_c = stem_channels
        for i, (in_mult, out_mult, n_blocks, stride) in enumerate(stage_cfg):
            out_c = int(base_channels * width_mul * out_mult)
            num_blocks = max(round(n_blocks * depth_mul), 1)
            
            stage = nn.Sequential(
                ConvModule(in_c, out_c, 3, stride, 1),
                CSPLayer(out_c, out_c, num_blocks=num_blocks)
            )
            self.stages.append(stage)
            in_c = out_c

    def forward(self, x):
        outs = []
        x = self.stem(x)
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if (i + 1) in self.out_indices:
                outs.append(x)
        return tuple(outs)

if __name__ == "__main__":
    model = CSPNeXt('s')
    x = torch.randn(1, 3, 640, 640)
    outs = model(x)
    for o in outs:
        print(o.shape)
