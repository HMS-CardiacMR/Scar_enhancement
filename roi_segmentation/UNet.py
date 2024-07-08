import torch
import monai
import torch.nn as nn

class SigmoidUNet(monai.networks.nets.UNet):
    def __init__(self, spatial_dims, in_channels, out_channels, channels, strides):
        super().__init__(spatial_dims, in_channels, out_channels, channels, strides)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Call the parent UNet's forward
        x = super().forward(x)
        # Apply softmax activation on the final layer
        x = self.sigmoid(x)
        return x