import torch # PyTorch package
import torch.nn as nn # basic building block for neural neteorks

VGG_config = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13-M': [64, 64, 'M', 128, 128, 'M', 256, 256, '1', 'M', 512, 512, '1', 'M', 512, 512, '1', 'M'],
    "VGG16" : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    "VGG19" : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}


class VGG(nn.Module):
    def __init__(self, num_channels, classes, arch):
        super(VGG, self).__init__() 
        self.features = self.create_features(arch=arch, num_channels=num_channels)

        # magic
        self.image_result = 256 // (2 ** arch.count('M'))
        self.avgpool = nn.AdaptiveAvgPool2d(8)

        self.classifier = nn.Sequential(
            nn.Linear(in_features=512 * 8 * 8, out_features=512),#(self.image_result ** 2)
            nn.BatchNorm1d(num_features=512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=512, out_features=256),
            nn.BatchNorm1d(num_features=256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=256, out_features=classes),
            nn.ReLU(),
            nn.LogSoftmax(dim=1)    
        )

    def create_features(self, arch, num_channels):
        layers = []
        _in = num_channels

        for x in arch:
            if type(x) == int:
                _out = x
                layers += self.create_conv_layer_group(_in, _out, (3,3))
                _in = x
            elif x == '1': 
                layers += self.create_conv_layer_group(_in, _out, (1,1), padding=(0,0))
            elif x == 'M':
                layers += [nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))]

        return nn.Sequential(*layers)

    def create_conv_layer_group(self, _in, _out, kernel, stride=(1,1), padding=(1,1)):
        layer = []

        conv = nn.Conv2d(in_channels=_in, out_channels=_out, kernel_size=kernel, stride=stride, padding=padding);
        nn.init.kaiming_uniform_(conv.weight, nonlinearity='relu', mode='fan_in')

        layer += [
            conv,
            nn.BatchNorm2d(num_features=_out),
            nn.ReLU()
        ]

        return layer

    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.shape[0], -1) 
        output = self.classifier(x)
        return output