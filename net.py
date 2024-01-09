import torch
import torch.nn as nn
import torch.nn.functional as F

def conv3x3(in_planes, out_planes):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=2):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BeginBlock(nn.Module):
    def __init__(self, inplanes=6, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.conv = conv3x3(1, inplanes)
        self.bn = norm_layer(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.maxpool(out)

        return out

class BasicBlock(nn.Module):
    def __init__(self, inplanes=6, planes=6, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out

class SimpleBlock(nn.Module):
    def __init__(self, inplanes=6, planes=6, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes // 2)
        self.bn1 = norm_layer(planes // 2)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes // 2, planes)
        self.bn2 = norm_layer(planes)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out

class SimpleBlock_NoBn(nn.Module):
    def __init__(self, inplanes=6, planes=6):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes // 2)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes // 2, planes)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.relu(out)

        out = self.conv2(out)

        out += identity
        out = self.relu(out)

        return out

class MLPBlock(nn.Module):
    def __init__(self, c=6, hw=196, c_dim=12, hw_dim=14):
        super().__init__()
        self.relu = nn.ReLU()
        #self.fc1_hw = nn.Linear(hw, hw_dim)
        #self.fc2_hw = nn.Linear(hw_dim, hw)
        self.fc1_c = nn.Linear(c, c_dim)
        self.fc2_c = nn.Linear(c_dim, c)

    def forward(self, x):
        B, C, H, W = x.shape
        identity = x

        out = x.reshape(B, C, H * W)

        #out = self.fc1_hw(out)
        #out = self.relu(out)
        #out = self.fc2_hw(out)

        out = out.transpose(1, 2)

        out = self.fc1_c(out)
        out = self.relu(out)
        out = self.fc2_c(out)

        out = out.transpose(1, 2)
        
        out = out.reshape(B, C, H, W)

        out += identity
        out = self.relu(out)

        return out

class AttnBlock(nn.Module):
    def __init__(self, in_ch=6):
        super().__init__()
        self.proj_q = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0, bias=False)
        self.proj_k = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0, bias=False)
        self.proj_v = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0, bias=False)
        self.proj = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0, bias=False)

    def forward(self, x):
        B, C, H, W = x.shape
        q = self.proj_q(x)
        k = self.proj_k(x)
        v = self.proj_v(x)

        q = q.permute(0, 2, 3, 1).view(B, H * W, C)
        k = k.view(B, C, H * W)
        w = torch.bmm(q, k) * (int(C) ** (-0.5))
        assert list(w.shape) == [B, H * W, H * W]
        w = F.softmax(w, dim=-1)

        v = v.permute(0, 2, 3, 1).view(B, H * W, C)
        h = torch.bmm(w, v)
        assert list(h.shape) == [B, H * W, C]
        h = h.view(B, H, W, C).permute(0, 3, 1, 2)
        h = self.proj(h)

        return x + h
    
class LastBlock(nn.Module):
    def __init__(self, inplanes=6, planes=12, stride=2, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.downsample = conv1x1(inplanes, planes, stride)
        self.bn = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(7*7*planes, 10)

    def forward(self, x):
        x = self.downsample(x)
        x = self.bn(x)
        x = self.relu(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x