

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


## ResNet핵심 (BasicBlock or Bottleneck)
class Bottleneck(nn.Module):
    def __init__(
        self
    ):
        super(Bottlenect, self).__init__()
        norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups # ?
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        # 1x1
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        # 3x3
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        # 1x1
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample()
        out += identity
        out = self.relu(out)
        return out



class ResNet(nn.Module):

    def __init__(
        self,
        block,
        layers,
        num_classes
    ):
        super(ResNet, self).__init__()
        norm_layer = nn.BatchNorm2d

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplane)
        self.relu = nn.ReLU(inplace=True)
        #..
        self.layer1 = self._make_layer()
        
    
    
    def _make_layer(self, block):
        layers = []
        layers.append(block(...))
        




    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        #..
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


        
