import torch.nn as nn
import torch
from torchvision.models import resnet152, inception_v3
from torch.nn.functional import interpolate

from .networks import Flatten, get_pad, GatedConv, GatCovnBlock, GatCovnWithAttention, Upsample, GatedResBlockWithAttention


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.layer1 = GatedConv(5, 32, 5, 1, padding=get_pad(256, 5, 1))
        self.layer2 = GatedConv(32, 64, 3, 2, padding=get_pad(256, 4, 2))
        self.layer3 = GatedConv(64, 64, 3, 1, padding=get_pad(128, 3, 1))
        self.layer4 = GatedConv(64, 128, 3, 2, padding=get_pad(128, 4, 2))
        self.layer5 = GatedConv(128, 128, 3, 1, padding=get_pad(64, 3, 1))
        self.layer6 = GatedConv(128, 128, 3, 1, padding=get_pad(64, 3, 1))
        self.layer7 = GatedConv(128, 128, 3, 1, dilation=2, padding=get_pad(64, 3, 1, 2))
        self.layer8 = GatedConv(128, 128, 3, 1, dilation=4, padding=get_pad(64, 3, 1, 4))
        self.layer9 = GatedConv(128, 128, 3, 1, dilation=8, padding=get_pad(64, 3, 1, 8))
        self.layer10 = GatedConv(128, 128, 3, 1, dilation=16, padding=get_pad(64, 3, 1, 16))
        self.layer11 = GatedConv(128, 128, 3, 1, padding=get_pad(64, 3, 1))
        self.layer12 = GatedConv(256, 128, 3, 1, padding=get_pad(64, 3, 1))
        self.layer13 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)  # Pixel shuffler is better?
        self.layer14 = GatedConv(256, 64, 3, 1, padding=get_pad(128, 3, 1))
        self.layer15 = GatedConv(128, 64, 3, 1, padding=get_pad(128, 3, 1))
        self.layer16 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)  # Pixel shuffler is better?
        self.layer17 = GatedConv(128, 32, 3, 1, padding=get_pad(256, 3, 1))
        self.layer18 = GatedConv(64, 16, 3, 1, padding=get_pad(256, 3, 1))
        self.layer19 = nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, img, prior, mask):
        input = torch.cat([img, prior], dim=1)
        noise = torch.rand(input.shape).cuda()
        input = torch.cat([input * (1 - mask) + noise * mask, mask], dim=1)
        out1 = self.layer1(input)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        out5 = self.layer5(out4)
        out = self.layer6(out5)
        out = self.layer7(out)
        out = self.layer8(out)
        out = self.layer9(out)
        out = self.layer10(out)
        out = self.layer11(out)
        out = torch.cat([out, out5], dim=1)
        out = self.layer12(out)
        out = torch.cat([out, out4], dim=1)
        out = self.layer13(out)
        out = self.layer14(out)
        out = torch.cat([out, out3], dim=1)
        out = self.layer15(out)
        out = torch.cat([out, out2], dim=1)
        out = self.layer16(out)
        out = self.layer17(out)
        out = torch.cat([out, out1], dim=1)
        out = self.layer18(out)
        out = self.layer19(out)
        out = out * mask + img * (1 - mask)
        return out

class GatedAttentionNet(nn.Module):
    def __init__(self, in_channels=5, out_channels=3, inner_channels=[64, 128, 256, 512], num_heads=[2, 2, 2, 2], atten_type="sparse",  last_covn=16, embed=4) -> None:
        super().__init__()
        self.encoder0 = GatCovnWithAttention(in_channels, inner_channels[0], 3, 2, 1, num_heads=num_heads[0], atten_type=atten_type)                  
        self.encoder1 = GatCovnWithAttention(inner_channels[0], inner_channels[1], 3, 2, 1, num_heads=num_heads[1], atten_type=atten_type)
        self.encoder2 = GatCovnWithAttention(inner_channels[1], inner_channels[2], 3, 2, 1, num_heads=num_heads[2], atten_type=atten_type)
        self.encoder3 = GatCovnWithAttention(inner_channels[2], inner_channels[3], 3, 2, 1, num_heads=num_heads[3], atten_type=atten_type)
        self.embedding = nn.Sequential()
        for i in range(embed):
            self.embedding.append(GatCovnWithAttention(inner_channels[-1], inner_channels[-1], 3, 1, 1, num_heads=num_heads[-1], atten_type="cc"))
        
        self.decoder0 = GatCovnWithAttention(inner_channels[-1], inner_channels[-2], 3, 1, 1, num_heads=num_heads[-1], atten_type=atten_type)
        self.decoder1 = GatCovnWithAttention(inner_channels[-2], inner_channels[-3], 3, 1, 1, num_heads=num_heads[-2], atten_type=atten_type)
        self.decoder2 = GatCovnWithAttention(inner_channels[-3], inner_channels[-4], 3, 1, 1, num_heads=num_heads[-3], atten_type=atten_type)
        self.decoder3 = GatCovnBlock(inner_channels[-4], last_covn, 3, 1, 1, num_heads=num_heads[-4], norm_groups=1)
        self.upsample0 = Upsample(inner_channels[-1])
        self.upsample1 = Upsample(inner_channels[-2])
        self.upsample2 = Upsample(inner_channels[-3])
        self.upsample3 = Upsample(inner_channels[-4])
        self.outcovn = nn.Conv2d(last_covn, out_channels, 3, 1, 1)
                                                                  
    def forward(self, img, prior, mask):
        x = torch.cat([img, prior], dim=1)
        noise = torch.randn(x.shape).to(img.device)
        x = torch.cat([x*(1-mask) + noise*mask, mask], dim=1)
        out1 = self.encoder0(x)
        out2 = self.encoder1(out1)
        out3 = self.encoder2(out2)
        out4 = self.encoder3(out3)
        out = self.embedding(out4)
        out = self.upsample0(out + out4)
        out = self.upsample1(self.decoder0(out) + out3)
        out = self.upsample2(self.decoder1(out) + out2)
        out = self.upsample3(self.decoder2(out) + out1)
        out = self.outcovn(self.decoder3(out))
        out = out * mask + img * (1- mask)
        return out

class GatedAttentionNetV2(nn.Module):
    def __init__(self, in_channels=5, out_channels=3, inner_channels=[64, 128, 256, 512], num_heads=[2, 2, 2, 2], atten_type="sparse",  last_covn=16, embed=4) -> None:
        super().__init__()
        self.encoder0 = GatCovnWithAttention(in_channels, inner_channels[0], 4, 2, 1, num_heads=num_heads[0], atten_type=atten_type)                  
        self.encoder1 = GatCovnWithAttention(inner_channels[0], inner_channels[1], 4, 2, 1, num_heads=num_heads[1], atten_type=atten_type)
        self.encoder2 = GatCovnWithAttention(inner_channels[1], inner_channels[2], 4, 2, 1, num_heads=num_heads[2], atten_type=atten_type)
        self.encoder3 = GatCovnWithAttention(inner_channels[2], inner_channels[3], 4, 2, 1, num_heads=num_heads[3], atten_type=atten_type)
        self.embedding = nn.Sequential()
        for i in range(embed):
            self.embedding.append(GatedResBlockWithAttention(inner_channels[-1], inner_channels[-1], 3, 1, 1, num_heads=num_heads[-1], atten_type="se"))
        
        self.decoder0 = GatCovnWithAttention(inner_channels[-1]*2, inner_channels[-2], 3, 1, 1, num_heads=num_heads[-1], atten_type=atten_type)
        self.decoder1 = GatCovnWithAttention(inner_channels[-2]*2, inner_channels[-3], 3, 1, 1, num_heads=num_heads[-2], atten_type=atten_type)
        self.decoder2 = GatCovnWithAttention(inner_channels[-3]*2, inner_channels[-4], 3, 1, 1, num_heads=num_heads[-3], atten_type=atten_type)
        self.decoder3 = GatCovnWithAttention(inner_channels[-4]*2, last_covn, 3, 1, 1, num_heads=num_heads[-4], atten_type=atten_type)
        self.upsample0 = Upsample(inner_channels[-2])
        self.upsample1 = Upsample(inner_channels[-3])
        self.upsample2 = Upsample(inner_channels[-4])
        self.upsample3 = Upsample(last_covn)
        self.outcovn = nn.Conv2d(last_covn, out_channels, 3, 1, 1)
                                                                  
    def forward(self, img, prior, mask):
        x = torch.cat([img*(1-mask), prior*(1-mask), mask-0.5], dim=1)
        # noise = torch.randn(x.shape).to(img.device)
        # x = torch.cat([x*(1-mask) + noise*mask, mask], dim=1)
        out1 = self.encoder0(x)
        out2 = self.encoder1(out1)
        out3 = self.encoder2(out2)
        out4 = self.encoder3(out3)
        out = self.embedding(out4)
        out = self.decoder0(torch.cat([out, out4], dim=1))
        out = self.upsample0(out)
        out = self.decoder1(torch.cat([out, out3], dim=1))
        out = self.upsample1(out)
        out = self.decoder2(torch.cat([out, out2], dim=1))
        out = self.upsample2(out)
        out = self.decoder3(torch.cat([out, out1], dim=1))
        out = self.upsample3(out)
        out = self.outcovn(out)
        out = out * mask + img * (1- mask)
        return out

class GatedAttentionNetV3(nn.Module):
    def __init__(self, in_channels=5, out_channels=3, inner_channels=[64, 128, 256, 512], num_heads=[2, 2, 2, 2], atten_type="sparse",  last_covn=32, embed=4) -> None:
        super().__init__()
        self.inconv = GatedConv(in_channels, 32, 7, 1, 3)
        self.encoder0 = GatCovnWithAttention(32, inner_channels[0], 5, 2, 2, num_heads=num_heads[0], atten_type=atten_type)                  
        self.encoder1 = GatCovnWithAttention(inner_channels[0], inner_channels[1], 5, 2, 2, num_heads=num_heads[1], atten_type=atten_type)
        self.encoder2 = GatCovnWithAttention(inner_channels[1], inner_channels[2], 4, 2, 1, num_heads=num_heads[2], atten_type=atten_type)
        self.encoder3 = GatCovnWithAttention(inner_channels[2], inner_channels[3], 4, 2, 1, num_heads=num_heads[3], atten_type=atten_type)
        self.embedding = nn.Sequential()
        for i in range(embed):
            self.embedding.append(GatedResBlockWithAttention(inner_channels[-1], inner_channels[-1], 3, 1, 1, num_heads=num_heads[-1], atten_type="se"))
        
        self.decoder0 = GatCovnWithAttention(inner_channels[-1]*2, inner_channels[-2], 3, 1, 1, num_heads=num_heads[-1], atten_type=atten_type)
        self.decoder1 = GatCovnWithAttention(inner_channels[-2]*2, inner_channels[-3], 3, 1, 1, num_heads=num_heads[-2], atten_type=atten_type)
        self.decoder2 = GatCovnWithAttention(inner_channels[-3]*2, inner_channels[-4], 5, 1, 2, num_heads=num_heads[-3], atten_type=atten_type)
        self.decoder3 = GatCovnWithAttention(inner_channels[-4]*2, last_covn, 5, 1, 2, num_heads=num_heads[-4], atten_type=atten_type)
        self.upsample0 = Upsample(inner_channels[-2], 7, 3)
        self.upsample1 = Upsample(inner_channels[-3], 7, 3)
        self.upsample2 = Upsample(inner_channels[-4], 7, 3)
        self.upsample3 = Upsample(last_covn)
        self.outcovn = nn.Conv2d(last_covn, out_channels, 7, 1, 3)
                                                                  
    def forward(self, img, prior, mask):
        x = torch.cat([img*(1-mask), prior*(1-mask), mask-0.5], dim=1)
        
        out1 = self.encoder0(self.inconv(x))
        out2 = self.encoder1(out1)
        out3 = self.encoder2(out2)
        out4 = self.encoder3(out3)
        out = self.embedding(out4)
        out = self.decoder0(torch.cat([out, out4], dim=1))
        out = self.upsample0(out)
        out = self.decoder1(torch.cat([out, out3], dim=1))
        out = self.upsample1(out)
        out = self.decoder2(torch.cat([out, out2], dim=1))
        out = self.upsample2(out)
        out = self.decoder3(torch.cat([out, out1], dim=1))
        out = self.upsample3(out)
        out = self.outcovn(out)
        out = out * mask + img * (1- mask)
        return out
    
# class GatedAttentionNetV3(nn.Module):
#     def __init__(self, in_channels=5, out_channels=3, inner_channels=[64, 128, 256, 512], num_heads=[2, 2, 2, 2], atten_type="sparse",  last_covn=32, embed=4) -> None:
#         super().__init__()
#         self.img_encoder0 = GatCovnWithAttention(in_channels, inner_channels[0], 4, 2, 1, num_heads=num_heads[0], atten_type=atten_type)                  
#         self.img_encoder1 = GatCovnWithAttention(inner_channels[0], inner_channels[1], 4, 2, 1, num_heads=num_heads[1], atten_type=atten_type)
#         self.img_encoder2 = GatCovnWithAttention(inner_channels[1], inner_channels[2], 4, 2, 1, num_heads=num_heads[2], atten_type=atten_type)
#         self.img_encoder3 = GatCovnWithAttention(inner_channels[2], inner_channels[3], 4, 2, 1, num_heads=num_heads[3], atten_type=atten_type)
#         self.cont_encoder0 = GatCovnWithAttention(in_channels, inner_channels[0], 4, 2, 1, num_heads=num_heads[0], atten_type=atten_type)                  
#         self.cont_encoder1 = GatCovnWithAttention(inner_channels[0], inner_channels[1], 4, 2, 1, num_heads=num_heads[1], atten_type=atten_type)
#         self.cont_encoder2 = GatCovnWithAttention(inner_channels[1], inner_channels[2], 4, 2, 1, num_heads=num_heads[2], atten_type=atten_type)
#         self.cont_encoder3 = GatCovnWithAttention(inner_channels[2], inner_channels[3], 4, 2, 1, num_heads=num_heads[3], atten_type=atten_type)
#         self.mix_conv = GatCovnWithAttention(inner_channels[3]*2, inner_channels[3], 3, 1, 1, num_heads=num_heads[3], atten_type=atten_type)
#         self.embedding = nn.Sequential()
#         for i in range(embed):
#             self.embedding.append(GatedResBlockWithAttention(inner_channels[-1], inner_channels[-1], 3, 1, 1, num_heads=num_heads[-1], atten_type="se"))
        
#         self.img_decoder0 = GatCovnWithAttention(inner_channels[-1]*2, inner_channels[-2], 3, 1, 1, num_heads=num_heads[-1], atten_type=atten_type)
#         self.img_decoder1 = GatCovnWithAttention(inner_channels[-2]*2, inner_channels[-3], 3, 1, 1, num_heads=num_heads[-2], atten_type=atten_type)
#         self.img_decoder2 = GatCovnWithAttention(inner_channels[-3]*2, inner_channels[-4], 3, 1, 1, num_heads=num_heads[-3], atten_type=atten_type)
#         self.img_decoder3 = GatCovnWithAttention(inner_channels[-4]*2, last_covn, 3, 1, 1, num_heads=num_heads[-4], atten_type=atten_type)
#         self.cont_decoder0 = GatCovnWithAttention(inner_channels[-1]*2, inner_channels[-2], 3, 1, 1, num_heads=num_heads[-1], atten_type=atten_type)
#         self.cont_decoder1 = GatCovnWithAttention(inner_channels[-2]*2, inner_channels[-3], 3, 1, 1, num_heads=num_heads[-2], atten_type=atten_type)
#         self.cont_decoder2 = GatCovnWithAttention(inner_channels[-3]*2, inner_channels[-4], 3, 1, 1, num_heads=num_heads[-3], atten_type=atten_type)
#         self.cont_decoder3 = GatCovnWithAttention(inner_channels[-4]*2, last_covn, 3, 1, 1, num_heads=num_heads[-4], atten_type=atten_type)
#         self.upsample0 = Upsample(inner_channels[-2])
#         self.upsample1 = Upsample(inner_channels[-3])
#         self.upsample2 = Upsample(inner_channels[-4])
#         self.img_upsample3 = Upsample(last_covn)
#         self.cont_upsample3 = Upsample(last_covn)
#         self.mix_conv0 = GatCovnWithAttention(inner_channels[-1], inner_channels[-2], 3, 1, 1, num_heads=num_heads[-1], atten_type=atten_type)
#         self.mix_conv1 = GatCovnWithAttention(inner_channels[-2], inner_channels[-3], 3, 1, 1, num_heads=num_heads[-1], atten_type=atten_type)
#         self.mix_conv2 = GatCovnWithAttention(inner_channels[-3], inner_channels[-4], 3, 1, 1, num_heads=num_heads[-1], atten_type=atten_type)
#         # self.mix_conv3 = GatCovnWithAttention(inner_channels[-4], last_covn, 3, 1, 1, num_heads=num_heads[-1], atten_type=atten_type)
#         self.img_outcovn = nn.Conv2d(last_covn, out_channels, 3, 1, 1)
#         self.cont_outcovn = nn.Conv2d(last_covn, 1, 3, 1, 1)
                                                                  
#     def forward(self, img, prior, mask):
#         x = torch.cat([img*(1-mask), prior*(1-mask), mask-0.5], dim=1)
#         # noise = torch.randn(x.shape).to(img.device)
#         # x = torch.cat([x*(1-mask) + noise*mask, mask], dim=1)
#         img_out1 = self.img_encoder0(x)
#         img_out2 = self.img_encoder1(img_out1)
#         img_out3 = self.img_encoder2(img_out2)
#         img_out4 = self.img_encoder3(img_out3)
#         cont_out1 = self.cont_encoder0(x)
#         cont_out2 = self.cont_encoder1(cont_out1)
#         cont_out3 = self.cont_encoder2(cont_out2)
#         cont_out4 = self.cont_encoder3(cont_out3)
#         out = self.embedding(self.mix_conv(torch.cat([img_out4, cont_out4], dim=1)))
#         out1 = self.img_decoder0(torch.cat([out, img_out4], dim=1))
#         out2 = self.cont_decoder0(torch.cat([out, cont_out4], dim=1))
#         out = self.mix_conv0(torch.cat([out1, out2], dim=1))
#         out = self.upsample0(out)
#         out1 = self.img_decoder1(torch.cat([out, img_out3], dim=1))
#         out2 = self.cont_decoder1(torch.cat([out, cont_out3], dim=1))
#         out = self.mix_conv1(torch.cat([out1, out2], dim=1))
#         out = self.upsample1(out)
#         out1 = self.img_decoder2(torch.cat([out, img_out2], dim=1))
#         out2 = self.cont_decoder2(torch.cat([out, cont_out2], dim=1))
#         out = self.mix_conv2(torch.cat([out1, out2], dim=1))
#         out = self.upsample2(out)
#         out1 = self.img_decoder3(torch.cat([out, img_out1], dim=1))
#         out2 = self.cont_decoder3(torch.cat([out, cont_out1], dim=1))
#         out1 = self.img_upsample3(out1)
#         out2 = self.cont_upsample3(out2)
#         out1 = self.img_outcovn(out1)
#         out2 = self.cont_outcovn(out2)
#         out1 = out1 * mask + img * (1- mask)
#         out2 = out2 * mask + prior * (1- mask)
#         return out1, out2

class Discriminator256(nn.Module):
    def __init__(self, in_channels):
        super(Discriminator256, self).__init__()

        self.layer1 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(in_channels, 64, kernel_size=5, stride=2, padding=2)),
            nn.LeakyReLU(),
        )
        self.layer2 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2)),
            nn.LeakyReLU(),
        )
        self.layer3 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2)),
            nn.LeakyReLU(),
        )
        self.layer4 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(256, 256, kernel_size=5, stride=2, padding=2)),
            nn.LeakyReLU(),
        )
        self.layer5 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(256, 256, kernel_size=5, stride=2, padding=2)),
            nn.LeakyReLU(),
        )
        self.layer6 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(256, 256, kernel_size=5, stride=2, padding=2)),
            nn.LeakyReLU(),
        )
        self.layer7 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(256, 256, kernel_size=4, stride=1, padding=0)),
            nn.LeakyReLU(),
        )
        self.layer8 = Flatten()
        # self.layer9 = nn.Linear(4096, 256, bias=False)
        # self.layer10 = nn.utils.spectral_norm(nn.Linear(1000, 256, bias=False))
        self.layer11 = nn.utils.spectral_norm(nn.Linear(256, 1, bias=False))

    def forward(self, x, y):
        out = torch.cat([x, y-0.5], dim=1)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)
        # out = self.layer9(out)
        out = self.layer11(out)

        return out

class Discriminator256V2(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(in_channels, 64, kernel_size=5, stride=2, padding=2)),
            nn.LeakyReLU(),
            nn.utils.spectral_norm(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)),
            nn.LeakyReLU(),
        )
        self.layer2 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2)),
            nn.LeakyReLU(),
            nn.utils.spectral_norm(nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)),
            nn.LeakyReLU(),
        )
        self.layer3 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2)),
            nn.LeakyReLU(),
            nn.utils.spectral_norm(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)),
            nn.LeakyReLU(),
        )
        self.layer4 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(256, 256, kernel_size=5, stride=2, padding=2)),
            nn.LeakyReLU(),
            nn.utils.spectral_norm(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)),
            nn.LeakyReLU(),
        )
        self.layer5 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(256, 256, kernel_size=5, stride=2, padding=2)),
            nn.LeakyReLU(),
        )
        self.layer6 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(256, 256, kernel_size=5, stride=2, padding=2)),
            nn.LeakyReLU(),
        )
        self.layer7 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(256, 256, kernel_size=4, stride=1, padding=0)),
            nn.LeakyReLU(),
        )
        self.layer8 = Flatten()
        # self.layer9 = nn.Linear(4096, 256, bias=False)
        # self.layer10 = nn.utils.spectral_norm(nn.Linear(1000, 256, bias=False))
        self.layer11 = nn.utils.spectral_norm(nn.Linear(256, 1, bias=False))

    def forward(self, x, y):
        out = torch.cat([x, y-0.5], dim=1)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)
        # out = self.layer9(out)
        out_t = self.layer11(out)

        return out_t

class Inception(nn.Module):
    def __init__(self):
        super().__init__()
        self.inception_v3 = inception_v3(weights='Inception_V3_Weights.DEFAULT', transform_input=True, aux_logits=False)
        self.inception_v3.fc = nn.Identity()
        
    def forward(self, x):
        x = interpolate(x, (299, 299), mode='bilinear', align_corners=False)
        # print(x.size())
        x = self.inception_v3(x)
        # print(type(x))
        # x = nn.functional.normalize(aux)
        return x


class Discriminator512(nn.Module):
    def __init__(self, in_channels=4):
        super(Discriminator512, self).__init__()

        self.layer1 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(in_channels, 64, kernel_size=5, stride=2, padding=2)),
            nn.LeakyReLU(),
        )
        self.layer2 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2)),
            nn.LeakyReLU(),
        )
        self.layer3 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2)),
            nn.LeakyReLU(),
        )
        self.layer4 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(256, 256, kernel_size=5, stride=2, padding=2)),
            nn.LeakyReLU(),
        )
        self.layer5 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(256, 256, kernel_size=5, stride=2, padding=2)),
            nn.LeakyReLU(),
        )
        self.layer6 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(256, 256, kernel_size=5, stride=2, padding=2)),
            nn.LeakyReLU(),
        )
        self.layer7 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(256, 256, kernel_size=5, stride=1, padding=0)),
            nn.LeakyReLU(),
        )
        self.layer8 = Flatten()
        self.layer9 = nn.Linear(4096, 256, bias=False)  # It might not be correct
        # self.layer10 = nn.Linear(1000, 256, bias=False)
        self.layer11 = nn.Linear(256, 1, bias=False)

    def forward(self, x, y):
        out = torch.cat([x, y-0.5], dim=1)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)
        out = self.layer9(out)
        out = self.layer11(out)

        # z = self.layer10(z)
        # out = (out * z).sum(1, keepdim=True)
        # out = torch.add(out, out_t)
        return out




