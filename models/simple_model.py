import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, c = (64, 32, 4)):
        super().__init__()
        
        c1, c2, c3 = c

        self.conv1 = nn.Conv2d(1, c1, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(c1)

        # downsampling -> 14 x 14
        self.conv2 = nn.Conv2d(c1, c2, 4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(c2)

        # downsampling -> 7 x 7
        self.conv3 = nn.Conv2d(c2, c3, 4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(c3)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        return x

class Decoder(nn.Module):
    def __init__(self, c = (32, 64), input_dim = 7 * 7 * 4):
        super().__init__()
        
        c1, c2 = c

        # upsampling * 4
        self.linear = torch.nn.Linear(input_dim, input_dim * 4 * 4)

        input_c = input_dim // (7 ** 2)
        self.conv1 = nn.ConvTranspose2d(input_c, c1, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(c1)

        self.conv2 = nn.ConvTranspose2d(c1, c2, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(c2)

        self.conv3 = nn.Conv2d(c2, 1, 1, 1)
        self.bn3 = nn.BatchNorm2d(1)

    def forward(self, x):
        N, C, H, W = x.shape
        reshaped = x.view(N, C * H * W)
        x = F.relu(self.linear(reshaped))
        x = x.view(N, C, H * 4, W * 4)
        
        # print("Hi", x.shape)
        x = F.relu(self.bn1(self.conv1(x)))
        # print(x.shape)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        return x
