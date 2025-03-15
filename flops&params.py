import torch
import torchvision
from model.NESS_Net import EdgeConstructModule
from thop import profile

# Model
print('==> Building model..')
model = EdgeConstructModule([128, 512, 1024],96)

dummy_input = torch.randn(1, 128, 384, 384)
# dummy_input1 = torch.randn(1, 256, 192, 192)
dummy_input2 = torch.randn(1, 512, 96, 96)
dummy_input3 = torch.randn(1, 1024, 48, 48)
# dummy_input4 = torch.randn(1, 1024, 48, 48)

flops, params = profile(model, (dummy_input, dummy_input2, dummy_input3))
print('flops: ', flops, 'params: ', params)
print('flops: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))
