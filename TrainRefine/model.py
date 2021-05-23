import torch
import torch.nn as nn
from TrainRefine.fcn import FCN_res18_FPN
from TrainRefine.icnnmodel import ICNNSegModel


class RefineModel(nn.Module):
    def __init__(self):
        super(RefineModel, self).__init__()
        self.model = FCN_res18_FPN(in_channels=4, pretrained=True)
        # self.model = ICNNSegModel(in_channels=4, out_channels=2)

    def forward(self, image, pred):
        # Image Shape (N, 3, H, W)
        # Pred Shape (N, 1, H, W)
        x = torch.cat([image, pred], dim=1)
        return self.model(x)

