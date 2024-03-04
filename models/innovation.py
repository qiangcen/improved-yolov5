import torch  # pytorch深度学习框架
import torch.nn as nn  # 专门为神经网络设计的模块化接口
from models.common import Conv


#===========================ISE module========================


class MP_Conv(nn.Module):
    def __init__(self, c1, c2):
        super(MP_Conv, self).__init__()
        self.m = nn.MaxPool2d(kernel_size=2, stride=2)
        self.cv1 = Conv(c1, c1//2, 1, 1)
        self.cv2 = Conv(c1//2, c2//2, 3, 2)
        self.cv3 = Conv(c1, c2//2, 1, 1)

    def forward(self, x):
        return torch.cat((self.cv3(self.m(x)), self.cv2(self.cv1(x))), 1)


class ISE(nn.Module):
    def __init__(self, c1, c2, r=16):
        super(ISE, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.l1 = nn.Linear(c2, c2 // r, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.l2 = nn.Linear(c2 // r, c2, bias=False)
        self.sig = nn.Sigmoid()
        self.mp_conv = MP_Conv(c1, c2)

    def forward(self, x):
        x = self.mp_conv(x)
        b, c, _, _ = x.size()
        y = self.avgpool(x).view(b, c)
        y = self.l1(y)
        y = self.relu(y)
        y = self.l2(y)
        y = self.sig(y)
        y = y.view(b, c, 1, 1)
        return x * y.expand_as(x)

#===========================ISE module========================


#===========================ConDown module========================


class ConDown_input1(nn.Module):
    def __init__(self, c1, c2):
        super(ConDown_input1, self).__init__()
        self.cv1 = Conv(4 * c1, c1, 1, 1)

        self.m1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.m2 = nn.MaxPool2d(kernel_size=4, stride=2, padding=1)
        self.m3 = nn.MaxPool2d(kernel_size=6, stride=2, padding=2)
        self.m4 = nn.MaxPool2d(kernel_size=8, stride=2, padding=3)

    def forward(self, x):
        return self.cv1(torch.cat((self.m1(x), self.m2(x), self.m3(x), self.m4(x)), 1))


class ConDown_input2(nn.Module):
    def __init__(self, c1, c2):
        super(ConDown_input2, self).__init__()
        self.m = nn.MaxPool2d(kernel_size=2, stride=2)
        self.a = nn.AvgPool2d(kernel_size=2, stride=2)
        self.cv1 = Conv(c1, c1 // 2, 1, 1)
        self.cv2 = Conv(2 * c1, c1 // 2, 1, 1)
        self.cv3 = Conv(c1, c1, 3, 2)

    def forward(self, x):
        return torch.cat((self.cv1(self.m(x)), self.cv2(torch.cat((self.a(x), self.cv3(x)), 1))), 1)

#===========================ConDown module========================
