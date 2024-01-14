"""SegmentationNN"""
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchvision import models



class SegmentationNN(nn.Module):
    def __init__(self, num_classes=23, hparams=None):
        super().__init__()
        # self.save_hyperparameters(hparams)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(256, 384, 1, 1, 0),
            nn.BatchNorm2d(384),
            nn.ReLU()
        )
        self.up2 = nn.ConvTranspose2d(384, 192, 5, 2,1)
        self.decoder2 = nn.Sequential(
            nn.Conv2d(384, 192, 1),
            nn.BatchNorm2d(192),
            nn.ReLU()
        )
        self.up1 = nn.ConvTranspose2d(192, 64, 5,2,1 )
        self.decoder2 = nn.Sequential(
            nn.Conv2d(384,192, 1),
            nn.BatchNorm2d(192),
            nn.ReLU()
        )
        self.up1 = nn.ConvTranspose2d(192, 64, 5,2 ,1)
        self.decoder1 = nn.Sequential(
            nn.Conv2d(128, 64,1),
            nn.BatchNorm2d(64),
            nn.ReLU()

        )
        self.up0 = nn.ConvTranspose2d(64,32,8, 4,0)
        self.out_conv = nn.Conv2d(32,23,1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode ='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight,1)
                nn.init.constant_(m.bias, 0)
    def forward(self,x):
        enc1=self.encoder1(x)
        down1 = self.down1(enc1)
        enc2 = self.encoder2(down1)
        down2 = self.down2(enc2)
        enc3 = self.encoder3(down2)

        bottleneck = self.bottleneck(enc3)

        up2 = self.up2(bottleneck)
        cat2 =torch.cat((up2, enc2), dim=1)
        dec2 = self.encoder2(cat2)

        up1 = self.up1(dec2)
        cat1 = torch.cat((up1, enc1), dim=1)
        dec1 = self.decoder1(cat1)

        up0 = self.up0(dec1)
        out_conv = self.out_conv(up0)
        return out_conv



    @property
    def is_cuda(self):
        """
        Check if model parameters are allocated on the GPU.
        """
        return next(self.parameters()).is_cuda

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)

        
class DummySegmentationModel(pl.LightningModule):

    def __init__(self, target_image):
        super().__init__()
        def _to_one_hot(y, num_classes):
            scatter_dim = len(y.size())
            y_tensor = y.view(*y.size(), -1)
            zeros = torch.zeros(*y.size(), num_classes, dtype=y.dtype)

            return zeros.scatter(scatter_dim, y_tensor, 1)

        target_image[target_image == -1] = 1

        self.prediction = _to_one_hot(target_image, 23).permute(2, 0, 1).unsqueeze(0)

    def forward(self, x):
        return self.prediction.float()
