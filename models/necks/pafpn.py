import torch
import torch.nn as nn
import torch.nn.functional as F
from ..layers.blocks import ConvModule, CSPLayer

class CSPNeXtPAFPN(nn.Module):
    """
    Path Aggregation Network with CSPNeXt blocks.
    
    Args:
        in_channels (list[int]): Input channels from backbone [P3, P4, P5]
        out_channels (int): Output channel for all scales
        num_csp_blocks (int): Number of blocks in CSP layer
        expand_ratio (float): Expansion ratio for CSP layer
    """
    def __init__(self, in_channels=[128, 256, 512], out_channels=128, num_csp_blocks=3, expand_ratio=0.5):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # P5 -> P4
        self.reduce_layers_top_down = nn.ModuleList()
        # P3 -> P4
        self.downsamples = nn.ModuleList()
        
        # Top-down connections
        self.top_down_blocks = nn.ModuleList()
        for i in range(len(in_channels) - 1, 0, -1):
            self.reduce_layers_top_down.append(
                ConvModule(in_channels[i], in_channels[i-1], 1, 1, 0)
            )
            self.top_down_blocks.append(
                CSPLayer(in_channels[i-1] + in_channels[i-1], in_channels[i-1], 
                         num_blocks=num_csp_blocks, expand_ratio=expand_ratio, act_cfg="SiLU")
            )
            
        # Bottom-up connections
        self.bottom_up_blocks = nn.ModuleList()
        for i in range(len(in_channels) - 1):
            self.downsamples.append(
                ConvModule(in_channels[i], in_channels[i], 3, 2, 1)
            )
            self.bottom_up_blocks.append(
                CSPLayer(in_channels[i] + in_channels[i+1], in_channels[i+1],
                         num_blocks=num_csp_blocks, expand_ratio=expand_ratio, act_cfg="SiLU")
            )
            
        # Output convolutions
        self.out_convs = nn.ModuleList()
        for i in range(len(in_channels)):
            self.out_convs.append(
                ConvModule(in_channels[i], out_channels, 3, 1, 1)
            )

    def forward(self, inputs):
        # inputs: [P3, P4, P5]
        assert len(inputs) == len(self.in_channels)
        
        # Top-down pathway
        inner_outs = [inputs[-1]]
        for idx in range(len(self.in_channels) - 1):
            feat_high = inner_outs[-1]
            feat_low = inputs[len(self.in_channels) - 2 - idx]
            
            feat_high_upsample = F.interpolate(self.reduce_layers_top_down[idx](feat_high), scale_factor=2.0)
            inner_outs.append(
                self.top_down_blocks[idx](torch.cat([feat_high_upsample, feat_low], dim=1))
            )
        
        # inner_outs is now [P5_inner, P4_inner, P3_inner]
        inner_outs = inner_outs[::-1] # [P3_inner, P4_inner, P5_inner]
        
        # Bottom-up pathway
        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_high = inner_outs[idx+1]
            
            feat_down = self.downsamples[idx](feat_low)
            outs.append(
                self.bottom_up_blocks[idx](torch.cat([feat_down, feat_high], dim=1))
            )
            
        # Output convolutions
        final_outs = []
        for i, conv in enumerate(self.out_convs):
            final_outs.append(conv(outs[i]))
            
        return tuple(final_outs)
