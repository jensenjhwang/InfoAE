import torch
import torch.nn as nn
import torch.nn.functional as F

class ResNetBlock(nn.Module):
    def __init__(self, c=64, k = 3, transpose=False):
        super().__init__()
        
        conv_layer = nn.Conv2d if not transpose else nn.ConvTranspose2d

        self.conv1 = conv_layer(c, c, k, stride=1, padding=(k-1)//2)
        self.bn1 = nn.BatchNorm2d(c)

        self.conv2 = conv_layer(c, c, k, stride=1, padding=(k-1)//2)
        self.bn2 = nn.BatchNorm2d(c)

    def forward(self, x):
        res = F.relu(self.bn1(self.conv1(x)))
        res = self.bn2(self.conv2(res))
        return F.relu(res + x)

class Encoder(nn.Module):
    def __init__(self, c = 64, final_dim=3):
        super().__init__()
        
        c2 = (c * 3) // 2
        c3 = (c2 * 3) // 2

        self.initial = nn.Conv2d(1, c, 1)
        self.bnd0 = nn.BatchNorm2d(c)

        self.res1 = ResNetBlock(c)
        self.res2 = ResNetBlock(c2)
        self.res3 = ResNetBlock(c3)

        self.down1 = nn.Conv2d(c, c2, 4, stride=2, padding=1) # increase channels but not by too much
        self.bnd1 = nn.BatchNorm2d(c2)

        self.down2 = nn.Conv2d(c2, c3, 4, stride=2, padding=1)
        self.bnd2 = nn.BatchNorm2d(c3)
        
        self.final = nn.Conv2d(c3, final_dim, 1)
        self.bnd3 = nn.BatchNorm2d(final_dim)

    def forward(self, x):
        x = F.relu(self.bnd0(self.initial(x)))
        x = self.res1(x)
        x = F.relu(self.bnd1(self.down1(x)))
        x = self.res2(x)
        x = F.relu(self.bnd2(self.down2(x)))
        x = self.res3(x)
        x = F.relu(self.bnd3(self.final(x)))
        return x

class Decoder(nn.Module):
    def __init__(self, c = 64, input_c = 3):
        super().__init__()
        
        c2 = (c * 3) // 2
        c3 = (c2 * 3) // 2

        self.initial = nn.ConvTranspose2d(input_c, c3, 1)
        self.bnu3 = nn.BatchNorm2d(c3)
        
        self.res3 = ResNetBlock(c3, transpose=True)
        self.res2 = ResNetBlock(c2, transpose=True)
        self.res1 = ResNetBlock(c, transpose=True)

        self.up2 = nn.ConvTranspose2d(c3, c2, 4, stride=2, padding=1)
        self.bnu2 = nn.BatchNorm2d(c2)
        self.up1 = nn.ConvTranspose2d(c2, c, 4, stride=2, padding=1)
        self.bnu1 = nn.BatchNorm2d(c)

        self.final = nn.Conv2d(c, 1, 1)
        self.bnu0 = nn.BatchNorm2d(1)
        # upsampling * 4
        # self.linear = torch.nn.Linear(input_dim, input_dim * 4 * 4)
        # self.scaler = torch.nn.Upsample(scale_factor=4)

        # input_c = input_dim // (7 ** 2)
        # self.conv1 = nn.ConvTranspose2d(input_c, c1, 3, stride=1, padding=1)
        # self.bn1 = nn.BatchNorm2d(c1)

        # self.conv2 = nn.ConvTranspose2d(c1, c2, 3, stride=1, padding=1)
        # self.bn2 = nn.BatchNorm2d(c2)

        # self.conv3 = nn.Conv2d(c2, 1, 1, 1)
        # self.bn3 = nn.BatchNorm2d(1)

    def forward(self, x):
        x = F.relu(self.bnu3(self.initial(x)))
        x = self.res3(x)
        x = F.relu(self.bnu2(self.up2(x)))
        x = self.res2(x)
        x = F.relu(self.bnu1(self.up1(x)))
        x = self.res1(x)
        x = F.relu(self.bnu0(self.final(x)))
        # # N, C, H, W = x.shape
        # # reshaped = x.view(N, C * H * W)
        # # x = F.relu(self.linear(reshaped))
        # # x = x.view(N, C, H * 4, W * 4)
        # x = self.scaler(x)

        # # print("Hi", x.shape)
        # x = F.relu(self.bn1(self.conv1(x)))
        # # print(x.shape)
        # x = F.relu(self.bn2(self.conv2(x)))
        # x = F.relu(self.bn3(self.conv3(x)))

        return x
