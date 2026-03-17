import sys
import torch
import collections.abc
import types

# Monkey patch for torch._six.container_abcs
if not hasattr(torch, '_six'):
    torch._six = types.ModuleType('torch._six')
    torch._six.container_abcs = collections.abc
    sys.modules['torch._six'] = torch._six

from model.NESS_Net import EdgeConstructModule
from thop import profile

# Model
print('==> Building model..')
# Based on NESS_Net.py: self.edge_layer = EdgeConstructModule([128, 256, 512, 1024, 1024], mid_fea=96)
model = EdgeConstructModule([128, 256, 512, 1024, 1024], mid_fea=96)

# Input size 384x384
# Swin Transformer outputs:
# x1: H/4, W/4, C=128
# x2: H/8, W/8, C=256
# x3: H/16, W/16, C=512
# x4: H/32, W/32, C=1024
# x5: Same as x4

dummy_input1 = torch.randn(1, 128, 96, 96)
dummy_input2 = torch.randn(1, 256, 48, 48)
dummy_input3 = torch.randn(1, 512, 24, 24)
dummy_input4 = torch.randn(1, 1024, 12, 12)
dummy_input5 = torch.randn(1, 1024, 12, 12)

flops, params = profile(model, inputs=(dummy_input1, dummy_input2, dummy_input3, dummy_input4, dummy_input5))
print('flops: ', flops, 'params: ', params)
print('flops: %.2f G, params: %.2f M' % (flops / 1e9, params / 1e6))
