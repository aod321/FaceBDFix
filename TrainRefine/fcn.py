import torch
from torch import nn
from torch.nn import functional as F
from collections import OrderedDict
from efficientnet_pytorch import EfficientNet
from TrainRefine.fpn import resnet_fpn_backbone


class FCNHead(nn.Sequential):
    def __init__(self, in_channels=256):
        layers = [
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups=2, num_channels=in_channels),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups=2, num_channels=in_channels),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels=in_channels, out_channels=2, kernel_size=1, stride=1, padding=0)
        ]

        super(FCNHead, self).__init__(*layers)


class FCN_res18_FPN(nn.Module):
    def __init__(self, in_channels=4, pretrained=True):
        super(FCN_res18_FPN, self).__init__()
        self.backbone = resnet_fpn_backbone(backbone_name='resnet18', pretrained=pretrained)
        self.backbone.body.conv1 = nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.backbone.eval()
        self.classifier = FCNHead(in_channels=256)

    def forward(self, x):
        input_shape = x.shape[-2:]
        # contract: features is a dict of tensors
        _, p = self.backbone(x)
        x = p["2"]
        x = self.classifier(x)
        return x


if __name__ == '__main__':
    model = FCN_res18_FPN(pretrained=True)
    print(model)
    input_test = torch.randn(1, 3, 512, 512)
    out_test = model(input_test)
    print(out_test.shape)

