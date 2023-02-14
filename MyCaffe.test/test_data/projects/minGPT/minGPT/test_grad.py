from re import X
import torch
import torch.nn as nn

import numpy as np
from test_base import DebugFunction
from layers import LinearEx

x_gradExpected = torch.tensor([[[[50.,70.,90.], [50.,70.,90.], [50.,70.,90.]],\
                                [[50.,70.,90.], [50.,70.,90.], [50.,70.,90.]],\
                                [[50.,70.,90.], [50.,70.,90.], [50.,70.,90.]],\
                                [[50.,70.,90.], [50.,70.,90.], [50.,70.,90.]]]])

gy = torch.tensor([[[[1.,1.,1. ], [1.,1.,1. ], [1.,1.,1. ]],\
                    [[1.,1.,1. ], [1.,1.,1. ], [1.,1.,1. ]],\
                    [[1.,1.,1. ], [1.,1.,1. ], [1.,1.,1. ]],\
                    [[1.,1.,1. ], [1.,1.,1. ], [1.,1.,1. ]]]])
x = torch.tensor([[[[1.,  2.,  3. ], [4.,  5.,  6. ], [7.,  8.,  9. ]],\
                   [[1.1, 2.1, 3.1], [4.1, 5.1, 6.1], [7.1, 8.1, 9.1]],\
                   [[1.2, 2.2, 3.2], [4.2, 5.2, 6.2], [7.2, 8.2, 9.2]],\
                   [[1.3, 2.3, 3.3], [4.3, 5.3, 6.3], [7.3, 8.3, 9.3]]]])
x.requires_grad = True;

wt = torch.tensor([[10.,20.,30.],[40.,50.,60.]])
wt_gradExpected = torch.tensor([[49.8,61.8,73.8],[49.8,61.8,73.8]])
wt.requires_grad = True

ip = LinearEx("ip", 3, 2, bias=False)
ip.weight.data = wt

y = ip(x)

sum2 = y.sum()
sum2.backward()

ip2 = nn.Linear(3, 2, bias=False)
ip2.weight.data = wt

x2 = x * 1
y2 = ip2(x2)

sum3 = y2.sum()
sum3.backward()


gx = torch.matmul(gy, wt)
xt = x.permute(0, 1, 3, 2)
gw = torch.matmul(xt, gy)


print(y.shape)
print(x.shape)
print(wt.shape)