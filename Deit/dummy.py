import torch

width = 20000
a = torch.rand(width, width).cuda()
b = torch.rand(width, width).cuda()
while True:
    a * b
