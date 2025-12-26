import torch
import torch.nn as nn
import torch.nn.functional as F

def get_activation(name="SiLU"):
    if name == "SiLU":
        return nn.SiLU(inplace=True)
    elif name == "ReLU":
        return nn.ReLU(inplace=True)
    return nn.SiLU(inplace=True)

class ConvModule(nn.Module):
    """A conv block that bundles conv/norm/activation."""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False, act_cfg="SiLU"):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = get_activation(act_cfg) if act_cfg else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class CSPNeXtBlock(nn.Module):
    """The basic block for CSPNeXt."""
    def __init__(self, in_channels, out_channels, expand_ratio=0.5, kernel_size=5, act_cfg="SiLU"):
        super().__init__()
        mid_channels = int(out_channels * expand_ratio)
        self.conv1 = ConvModule(in_channels, mid_channels, 3, 1, 1, act_cfg=act_cfg)
        self.conv2 = ConvModule(mid_channels, out_channels, kernel_size, 1, kernel_size//2, groups=mid_channels, act_cfg=act_cfg)
        # Shortcut (Identity) if dimensions match
        self.add_identity = (in_channels == out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        if self.add_identity:
            return x + out
        return out

class CSPLayer(nn.Module):
    """CSP Layer usually consisting of a few blocks."""
    def __init__(self, in_channels, out_channels, num_blocks=1, expand_ratio=0.5, act_cfg="SiLU"):
        super().__init__()
        mid_channels = int(out_channels * expand_ratio)
        self.main_conv = ConvModule(in_channels, mid_channels, 1, 1, 0, act_cfg=act_cfg)
        self.short_conv = ConvModule(in_channels, mid_channels, 1, 1, 0, act_cfg=act_cfg)
        
        self.blocks = nn.Sequential(*[
            CSPNeXtBlock(mid_channels, mid_channels, 1.0, 5, act_cfg=act_cfg)
            for _ in range(num_blocks)
        ])
        
        self.final_conv = ConvModule(mid_channels * 2, out_channels, 1, 1, 0, act_cfg=act_cfg)

    def forward(self, x):
        x_short = self.short_conv(x)
        x_main = self.main_conv(x)
        x_main = self.blocks(x_main)
        return self.final_conv(torch.cat((x_main, x_short), dim=1))
