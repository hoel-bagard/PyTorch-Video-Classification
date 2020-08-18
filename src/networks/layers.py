import torch
import torch.nn as nn
import torch.nn.functional as F


class DarknetConv(torch.nn.Module):
    def __init__(self, in_filters, out_filters, size, stride=1, padding=0):
        super(DarknetConv, self).__init__()
        self.conv = nn.Conv2d(in_filters, out_filters, kernel_size=size,
                              stride=stride, padding=padding, bias=False)
        self.batch_norm = nn.BatchNorm2d(out_filters)

    def forward(self, x):
        x = self.conv(x)
        x = self.batch_norm(x)
        x = F.leaky_relu(x, 0.1)

        return x


class DarknetResidualBlock(nn.Module):
    def __init__(self, filters):
        super(DarknetResidualBlock, self).__init__()
        self.darknet_conv1 = DarknetConv(filters, filters//2, 1)
        self.darknet_conv2 = DarknetConv(filters//2, filters, 3, padding=1)

    def forward(self, inputs):
        x = self.darknet_conv1(inputs)
        x = self.darknet_conv2(x)
        x = torch.add(inputs, x)
        return x


class DarknetBlock(nn.Module):
    def __init__(self, in_filters, out_filters, blocks):
        super(DarknetBlock, self).__init__()
        self.dark_conv = DarknetConv(in_filters, out_filters, 3, stride=2)
        self.dark_res_blocks = nn.Sequential(*[DarknetResidualBlock(out_filters) for _ in range(blocks)])

    def forward(self, inputs):
        x = self.dark_conv(inputs)
        for dark_res_block in self.dark_res_blocks:
            x = dark_res_block(x)
        return x
