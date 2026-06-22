import torch
from torch import nn

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    def forward(self, x):
        return self.conv(x)    
    

# Dépendamment de l'efficacité du modèle, essayer 3 et 5 niveau de profondeur au lieu de 4
class U_net(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        self.down1 = DoubleConv(in_channels, 64)
        self.down2 = DoubleConv(64, 128)
        self.down3 = DoubleConv(128, 256)
        self.down4 = DoubleConv(256, 512)
        self.bottleneck = DoubleConv(512, 1024)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.upconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.upconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.upconv4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up1 = DoubleConv(1024, 512)
        self.up2 = DoubleConv(512, 256)
        self.up3 = DoubleConv(256, 128)
        self.up4 = DoubleConv(128, 64)
        self.final = nn.Conv2d(64, 1, kernel_size=1)


# Pour simplifier et faire les tests/ajustements je garde ça tel quel
# Par contre en version final ça va être storé en liste
    def forward(self, x):
        s1 = self.down1(x)
        x = self.pool(s1)
        s2 = self.down2(x)
        x = self.pool(s2)
        s3 = self.down3(x)
        x = self.pool(s3)
        s4 = self.down4(x)
        x = self.pool(s4)
        x = self.bottleneck(x)
        x = self.upconv1(x)
        x = torch.cat([x, s4], dim=1)
        x = self.up1(x)
        x = self.upconv2(x)
        x = torch.cat([x, s3], dim=1)
        x = self.up2(x)
        x = self.upconv3(x)
        x = torch.cat([x, s2], dim=1)
        x = self.up3(x)
        x = self.upconv4(x)
        x = torch.cat([x, s1], dim=1)
        x = self.up4(x)
        return self.final(x)
