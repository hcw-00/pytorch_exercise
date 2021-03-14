# nn.Module
# nn.ModuleDict

cfgs = {
    'A':[64, 'M', 128, 128, 'M', 256]
}

class VGG(nn.Module):
    def __init__(self, features, num_classes=4, init_weights=True):
        super(VGG, self).__init__()
        self.feature = features
        self.avgpool = nn.AdaptiveAvgPool2d((3,3))
        self.classifier = nn.Sequential(
            nn.Linear(),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(),
            nn.Sigmoid()
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.feature(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

# VGG module
def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
        in_channels = v
    return nn.Sequential(*layers)


