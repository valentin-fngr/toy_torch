import torch.nn as nn 


class ConvModule(nn.Module): 

    """
    A simple conv -> bn -> relu block 
    """

    def __init__(
        self, 
        in_c, 
        out_c, 
        kernel_size=3,
        stride=1, 
        padding=0
    ): 
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU()


    def forward(self, x): 
        out = self.conv(x) 
        out = self.bn(out) 
        out = self.relu(out)

        return out


class ResnetBlock2n(nn.Module): 
    """
    Simple Resnet layer with 2*2 blocks
    block1 (reduce dimension or not) -> block2 -> block3 -> block4
    """
    def __init__(self, in_c, out_c, reduce_dim=False): 
        super().__init__()

        self.reduce_dim = True
        stride = 2 if reduce_dim else 1 
        self.conv1 = ConvModule(in_c, out_c, 3, stride=stride, padding=1)
        self.conv2 = ConvModule(out_c, out_c, 3, 1, 1) 
        self.conv3 = ConvModule(out_c, out_c, 3, 1, 1) 
        self.conv4 = ConvModule(out_c, out_c, 3, 1, 1) 

        self.shortcut = nn.Conv2d(in_c, out_c, 1, stride=stride, padding=0)

    def forward(self, x): 
        
        identity = self.shortcut(x)
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        out = x2 + identity
        x3 = self.conv3(out)
        x4 = self.conv4(x3)
        out = x4 + out 
        
        return out
        


class ResnetBlock3n(ResnetBlock2n):
    """
    Simple Resnet layer with 3*2 blocks. 
    This class extends the above ResnetBlock2n class
    """
    def __init__(self, in_c, out_c, reduce_dim=False): 
        super().__init__(in_c, out_c, reduce_dim)

        self.conv5 = ConvModule(out_c, out_c, 3, 1, 1)
        self.conv6 = ConvModule(out_c, out_c, 3, 1, 1) 

    def forward(self, x): 
        x4 = super().forward(x) 
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)

        return x6



class DepthWiseSeperableConvolution(nn.Module): 

    def __init__(
        self,
        in_c, 
        out_c, 
        kernel_size=3, 
        stride=1, 
        padding=0 
    ):
        super().__init__()
        self.dwconv = nn.Conv2d(in_c, in_c, kernel_size, stride, padding, bias=False)
        self.bn1 = nn.BatchNorm2d(in_c)
        self.relu1 = nn.ReLU()

        self.pwconv = nn.Conv2d(in_c, out_c, 1, 1, 0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.relu2 = nn.ReLU()

    def forward(self, x): 

        x1 = self.dwconv(x) 
        x2 = self.bn1(x1)
        x3 = self.relu1(x2)

        x4 = self.pwconv(x3)
        x5 = self.bn2(x4)
        x6 = self.relu2(x5)
        return x6

