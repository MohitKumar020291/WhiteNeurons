import torch
import torch.nn as nn
import torch.nn.functional as F



# Convolutional Block: Conv2D -> BatchNorm -> ReLU (x2)
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),  # not in original U-Net, but added here as in Keras
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),  # not in original U-Net, but added here as in Keras
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


# Encoder Block: ConvBlock -> MaxPooling
class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EncoderBlock, self).__init__()
        self.conv = ConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        p = self.pool(x)
        return x, p  # x is skip connection, p is pooled down


# Decoder Block: Upsample -> Concat -> ConvBlock
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels, out_channels)  # note: in_channels includes skip connection

    def forward(self, x, skip):
        x = self.up(x)
        
        # Pad x if needed to match skip size (handling odd input sizes)
        if x.shape != skip.shape:
            x = F.pad(x, self._get_padding(x, skip))

        x = torch.cat([x, skip], dim=1)  # concatenate along channel dimension
        return self.conv(x)

    def _get_padding(self, x, skip):
        diffY = skip.size()[2] - x.size()[2]
        diffX = skip.size()[3] - x.size()[3]
        return [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2]


# U-Net Model
class UNet(nn.Module):
    def __init__(self, input_channels=3, output_channels=1):
        super(UNet, self).__init__()
        
        self.enc1 = EncoderBlock(input_channels, 6)
        self.enc2 = EncoderBlock(6, 12)
        self.enc3 = EncoderBlock(12, 24)
        self.enc4 = EncoderBlock(24, 48)

        self.bottleneck = ConvBlock(48, 96)

        self.dec1 = DecoderBlock(96, 48)
        self.dec2 = DecoderBlock(48, 24)
        self.dec3 = DecoderBlock(24, 12)
        self.dec4 = DecoderBlock(12, 6)

        self.final_conv = nn.Conv2d(6, output_channels, kernel_size=1)

    def forward(self, x):
        s1, p1 = self.enc1(x)
        s2, p2 = self.enc2(p1)
        s3, p3 = self.enc3(p2)
        s4, p4 = self.enc4(p3)

        b = self.bottleneck(p4)

        d1 = self.dec1(b, s4)
        d2 = self.dec2(d1, s3)
        d3 = self.dec3(d2, s2)
        d4 = self.dec4(d3, s1)

        return torch.sigmoid(self.final_conv(d4))  # for binary segmentation