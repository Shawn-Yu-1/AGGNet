import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import math


def get_pad(in_, ksize, stride, atrous=1):
    out_ = np.ceil(float(in_) / stride)
    return int(((out_ - 1) * stride + atrous * (ksize - 1) + 1 - in_) / 2)


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
    
def l2normalize(v, eps = 1e-12):
    return v / (v.norm() + eps)

# init network
def init_weights(model, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight.data, gain=gain)

            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
                
        elif classname.find('BatchNorm2d') != -1:
            nn.init.normal_(m.weight.data, 1.0, gain)
            nn.init.constant_(m.bias.data, 0.0)
    init_func(model)


class GatedConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,):
        super(GatedConv, self).__init__()
        self.conv2d = nn.utils.spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias))
        self.mask_conv2d = nn.utils.spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias))
        self.sigmoid = torch.nn.Sigmoid()
        
    def gated(self, mask):
        return self.sigmoid(mask)

    def forward(self, input):
        x = self.conv2d(input)
        mask = self.mask_conv2d(input)
        x = F.instance_norm(x * self.gated(mask))
        x = F.leaky_relu(x)
        return x

class SelfAttention(nn.Module):
    def __init__(self, in_channel, n_head=1, norm_groups=32):
        super().__init__()

        self.n_head = n_head

        self.norm = nn.GroupNorm(norm_groups, in_channel)
        self.qkv = nn.Conv2d(in_channel, in_channel * 3, 1, bias=False)
        self.out = nn.Conv2d(in_channel, in_channel, 1)
        
    def forward(self, input):
        batch, channel, height, width = input.shape
        n_head = self.n_head
        head_dim = channel // n_head

        norm = self.norm(input)
        qkv = self.qkv(norm).view(batch, n_head, head_dim * 3, height, width)
        query, key, value = qkv.chunk(3, dim=2)  # bhdyx

        attn = torch.einsum("bnchw, bncyx -> bnhwyx", query, key).contiguous() / math.sqrt(channel)
        attn = attn.view(batch, n_head, height, width, -1)
        attn = torch.softmax(attn, -1)
        attn = attn.view(batch, n_head, height, width, height, width)

        out = torch.einsum("bnhwyx, bncyx -> bnchw", attn, value).contiguous()
        out = self.out(out.view(batch, channel, height, width))

        return out + input

# gated attention block
class GatCovnWithAttention(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, num_heads=1, norm_groups = 32, atten_type="self") -> None:
        super().__init__()
        self.conv1 = GatedConv(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.conv2 = GatedConv(out_channels, out_channels, 3, 1, 1, dilation, groups, bias)
        if atten_type == "self":
            self.atten = SelfAttention(out_channels, num_heads, norm_groups)
        elif atten_type == "sparse":
            self.atten = SpatialAttention(out_channels)
        elif atten_type == "cc":
            self.atten = CrissCrossAttention(out_channels)
        else:
            ValueError("attention type is not in list!")
    
    def forward(self, x):
        x = self.conv2(self.conv1(x))
        out = self.atten(x)
        return out
    
# pure gatedcovn block  
class GatCovnBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, num_heads=1, norm_groups = 32) -> None:
        super().__init__()
        self.conv1 = GatedConv(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.conv2 = GatedConv(out_channels, out_channels, 3, 1, 1, dilation, groups, bias)
    
    def forward(self, x):
        out = self.conv2(self.conv1(x))
        return out 

# gated conv with residual connect
class GatedResBlockWithAttention(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, num_heads=1, norm_groups = 32, atten_type="self") -> None:
        super().__init__()
        self.conv1 = GatedConv(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.conv2 = GatedConv(out_channels, out_channels, 3, 1, 1, dilation, groups, bias)
        if atten_type == "self":
            self.atten = SelfAttention(out_channels, num_heads, norm_groups)
        elif atten_type == "sparse":
            self.atten = SpatialAttention(out_channels)
        elif atten_type == "cc":
            self.atten = CrissCrossAttention(out_channels)
        elif atten_type == "se":
            self.atten = SeAttention(out_channels)
        else:
            ValueError("attention type is not in list!")
    
    def forward(self, x):
        xx = self.conv2(self.conv1(x)) + x
        out = self.atten(xx)
        return out
    
class Upsample(nn.Module):
    """upsample the feature to 2x

    Args:
        in_channels: the input channels
    """
    def __init__(self, in_channels, conv_kernal=3, pad=1):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Conv2d(in_channels, in_channels, conv_kernal, padding=pad)
        self.se = SeAttention(in_channels)

    def forward(self, x):
        return self.se(F.leaky_relu(self.conv(self.up(x))))
    
# sparse attention 
class LargeKernelAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(
            dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)
        return u * attn

class SpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.dim = in_channels
        self.proj_1 = nn.Conv2d(in_channels, in_channels, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = LargeKernelAttention(in_channels)
        self.proj_2 = nn.Conv2d(in_channels, in_channels, 1)

    def forward(self, x):
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shorcut
        return x
    
# SEattention from MobileNetv3
class SeAttention(nn.Module):
    def __init__(self, in_size, reduction=4):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_size, in_size // reduction, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_size // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_size // reduction, in_size, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_size),
            hsigmoid()
        )

    def forward(self, x):
        return x * self.se(x)

class hsigmoid(nn.Module):
    def forward(self, x):
        out = F.relu6(x + 3, inplace=True) / 6
        return out

#  criss-cross Attention 
#  This code is borrowed from Serge-weihao/CCNet-Pure-Pytorch

def INF(B,H,W):
     return -torch.diag(torch.tensor(float("inf")).cuda().repeat(H),0).unsqueeze(0).repeat(B*W,1,1)


class CrissCrossAttention(nn.Module):
    """ Criss-Cross Attention Module"""
    def __init__(self, in_dim):
        super(CrissCrossAttention,self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax = nn.Softmax(dim=3)
        self.INF = INF
        self.gamma = nn.Parameter(torch.zeros(1))


    def forward(self, x):
        m_batchsize, _, height, width = x.size()
        proj_query = self.query_conv(x)
        proj_query_H = proj_query.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height).permute(0, 2, 1)
        proj_query_W = proj_query.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width).permute(0, 2, 1)
        proj_key = self.key_conv(x)
        proj_key_H = proj_key.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height)
        proj_key_W = proj_key.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width)
        proj_value = self.value_conv(x)
        proj_value_H = proj_value.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height)
        proj_value_W = proj_value.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width)
        energy_H = (torch.bmm(proj_query_H, proj_key_H)+self.INF(m_batchsize, height, width)).view(m_batchsize,width,height,height).permute(0,2,1,3)
        energy_W = torch.bmm(proj_query_W, proj_key_W).view(m_batchsize,height,width,width)
        concate = self.softmax(torch.cat([energy_H, energy_W], 3))

        att_H = concate[:,:,:,0:height].permute(0,2,1,3).contiguous().view(m_batchsize*width,height,height)
        #print(concate)
        #print(att_H) 
        att_W = concate[:,:,:,height:height+width].contiguous().view(m_batchsize*height,width,width)
        out_H = torch.bmm(proj_value_H, att_H.permute(0, 2, 1)).view(m_batchsize,width,-1,height).permute(0,2,3,1)
        out_W = torch.bmm(proj_value_W, att_W.permute(0, 2, 1)).view(m_batchsize,height,-1,width).permute(0,2,1,3)
        #print(out_H.size(),out_W.size())
        return self.gamma*(out_H + out_W) + x


