import numpy as np
import torch
import torch.nn.functional as F
# ckpt_file = "pretrained/yolox_s.pth"
# ckpt = torch.load(ckpt_file)['model']
# for k in list(ckpt.keys()):
#     if 'head.cls_preds' in k:
#         print(k)
#         l = ckpt[k].shape
#         print(l)

# x = np.empty((0, 5))
# print(x.shape)
# x= np.ones((7,10))
# x[:, 7] += 1
# print(x[:, 2:])

# x = torch.ones((7,3,2)).sigmoid_()
# y = torch.ones((7,3,2)).sigmoid_()
# z = F.binary_cross_entropy(x, y)
# print(z)
# x = torch.rand((3,128))
# y = x.reshape(-1).reshape((3,128))
# print(x == y)
x = torch.tensor([[0.02,0.5,0.0],[0.3,0.8,0.1]])
y = torch.tensor([[0.02,0.322,1.0],[0.0,1.0,0.0]])
z=F.binary_cross_entropy(x,y,reduction='none')
print(x.shape == y.shape)
# x = torch.tensor([0.0011,2,3])
# x.bool()
# print(x.bool())

