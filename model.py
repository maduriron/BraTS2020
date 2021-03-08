import torch

import torch.nn as nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint, checkpoint_sequential


class RSNBlock(nn.Module):
    def __init__(self, inplanes, outplanes,
               strides=[1, 1], dilations=[1, 1]):
        super(RSNBlock, self).__init__()
        self.groupNorm1 = nn.GroupNorm(num_groups=8,
                                       num_channels=inplanes)
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv3d(in_channels=inplanes,
                               out_channels=outplanes,
                               kernel_size=3,
                               stride=strides[0],
                               padding=dilations[0],
                               dilation=dilations[0])

        self.groupNorm2 = nn.GroupNorm(num_groups=8,
                                      num_channels=outplanes)
        self.conv2 = nn.Conv3d(in_channels=outplanes, 
                              out_channels=outplanes,
                              kernel_size=3,
                              stride=strides[1],
                              padding=dilations[1], 
                              dilation=dilations[1])

    def forward(self, x):
        residual = x
        features = self.groupNorm1(x)
        features = self.relu(x)
        features = self.conv1(features)

        features = self.groupNorm2(features)
        features = self.relu(features)
        features = self.conv2(features)
        out_res = residual + features

        return out_res

class DownSample(nn.Module):
    def __init__(self, inplanes, outplanes):
        super(DownSample, self).__init__()
        self.conv1 = nn.Conv3d(in_channels=inplanes,
                              out_channels=outplanes,
                              kernel_size=3,
                              stride=2, padding=1)
    def forward(self, x):
        x = self.conv1(x)
        return x
    


class Encoder(nn.Module):
    def __init__(self, inplanes, channels=32):
        super(Encoder, self).__init__()
        self.initial_conv = nn.Conv3d(in_channels=inplanes,
                                      out_channels=channels,
                                      kernel_size=3,
                                      stride=1, padding=1)
        self.drop = nn.Dropout(0.2)
        self.block1 = RSNBlock(inplanes=channels, outplanes=channels)
        self.downsample1 = DownSample(channels, channels*2)

        self.block2 = RSNBlock(inplanes=channels*2, outplanes=channels*2)
        self.block3 = RSNBlock(inplanes=channels*2, outplanes=channels*2)

        self.downsample2 = DownSample(channels*2, channels*4)

        self.block4 = RSNBlock(inplanes=channels*4, outplanes=channels*4)
        self.block5 = RSNBlock(inplanes=channels*4, outplanes=channels*4,
                               strides=[1, 1], dilations=[2, 2])

        self.downsample3 = DownSample(channels*4, channels*8)

        self.block6 = RSNBlock(inplanes=channels*8, outplanes=channels*8)
        self.block7 = RSNBlock(inplanes=channels*8, outplanes=channels*8,
                               strides=[1, 1], dilations=[4, 4])
        self.block8 = RSNBlock(inplanes=channels*8, outplanes=channels*8,
                               strides=[1, 1], dilations=[8, 8])
        self.block9 = RSNBlock(inplanes=channels*8, outplanes=channels*8,
                               strides=[1, 1], dilations=[16, 16])

    def forward(self, x):

        x = self.initial_conv(x)
        x = self.drop(x)

        x_1 = self.block1(x) 
        x_1_down = self.downsample1(x_1)

        x_2 = self.block2(x_1_down)
        x_3 = self.block3(x_2)  
        x_3_down = self.downsample2(x_3)

        x_4 = self.block4(x_3_down)
        x_5 = self.block5(x_4) 
        x_5_down = self.downsample3(x_5)

        x_6 = self.block6(x_5_down)
        x_7 = self.block7(x_6)
        x_8 = self.block8(x_7)
        x_9 = self.block9(x_8)

        return x_1, x_3, x_9
    
class ASPPModule(nn.Module):
    def __init__(self, inplanes, outplanes, dilation):
        super(ASPPModule, self).__init__()
        self.groupNorm = nn.GroupNorm(num_groups=8,
                                      num_channels=outplanes)
        self.relu = nn.ReLU()
        self.conv = nn.Conv3d(in_channels=inplanes,
                              out_channels=outplanes,
                              kernel_size=3,
                              stride=1, dilation=dilation,
                              padding=dilation)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.groupNorm(x)
        x = self.relu(x)

        return x
    
class ASPP(nn.Module):
    def __init__(self, inplanes, outplanes):
        super(ASPP, self).__init__()
        self.conv1 = nn.Conv3d(in_channels=inplanes, out_channels=outplanes, kernel_size=1, stride=1)
        self.aspp1 = ASPPModule(inplanes, outplanes, dilation=6)
        self.aspp2 = ASPPModule(inplanes, outplanes, dilation=10)
        self.aspp3 = ASPPModule(inplanes, outplanes, dilation=12)
        self.global_avg_pooling = nn.Sequential(nn.AdaptiveAvgPool3d((1, 1, 1)),
                                    nn.Conv3d(in_channels=inplanes, out_channels=outplanes, 
                                              kernel_size=1, stride=1),
                                    nn.GroupNorm(num_groups=8,
                                              num_channels=outplanes),
                                    nn.ReLU())
        self.conv2 = nn.Conv3d(in_channels=outplanes*5, out_channels=outplanes,
                               kernel_size=1)
        self.groupNorm1 = nn.GroupNorm(num_groups=8,
                                   num_channels=outplanes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.aspp1(x)
        x3 = self.aspp2(x)
        x4 = self.aspp3(x)
        x5 = self.global_avg_pooling(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], align_corners=True, mode='trilinear')

        pyramid_features = torch.cat([x1, x2, x3, x4, x5], dim=1)

        output = self.conv2(pyramid_features)
        output = self.groupNorm1(output)
        output = self.relu(output)

        return output


class UpSampleDeconv(nn.Module):
    def __init__(self, inplanes, outplanes, scale):
        super(UpSampleDeconv, self).__init__()
        self.deconv1 = nn.ConvTranspose3d(in_channels=inplanes,
                                         out_channels=outplanes,
                                         kernel_size=scale,
                                         stride=scale)
    def forward(self, x):
        x = self.deconv1(x)
        return x
    
class UpSampleInterpolation(nn.Module):
    def __init__(self, inplanes, outplanes, scale):
        super(UpSampleInterpolation, self).__init__()
        self.conv = nn.Conv3d(in_channels=inplanes,
                             out_channels=outplanes,
                             kernel_size=1, stride=1)
        self.scale = scale
        
    def forward(self, x):
        x = self.conv(x)
        x = F.interpolate(x, scale_factor=self.scale, align_corners=True, mode='trilinear')
        return x
    
class DecoderDeconv(nn.Module):
    def __init__(self, inplanes, outplanes):
        super(DecoderDeconv, self).__init__()
        self.upsample1 = UpSampleDeconv(inplanes, inplanes, scale=4)
        self.upsample2 = UpSampleDeconv(inplanes, inplanes // 2, scale=2)
        self.final_conv = nn.Conv3d(in_channels=inplanes // 2, 
                                    out_channels=outplanes,
                                    kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_low_1, x_low_2, pyramid):
        pyramid_up = self.upsample1(pyramid)
        output = pyramid_up + x_low_2
        output = self.upsample2(output)
        output = output + x_low_1
        output = self.final_conv(output)
        output = self.sigmoid(output)

        return output
    
class DecoderInterpolation(nn.Module):
    def __init__(self, inplanes, outplanes):
        super(DecoderInterpolation, self).__init__()
        self.upsample1 = UpSampleInterpolation(inplanes, inplanes, scale=4)
        self.upsample2 = UpSampleInterpolation(inplanes, inplanes // 2, scale=2)
        self.final_conv = nn.Conv3d(in_channels=inplanes // 2, 
                                    out_channels=outplanes,
                                    kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_low_1, x_low_2, pyramid):
        pyramid_up = self.upsample1(pyramid)
        output = pyramid_up + x_low_2
        output = self.upsample2(output)
        output = output + x_low_1
        output = self.final_conv(output)
        output = self.sigmoid(output)

        return output
    
class ContourAwareDecoder(nn.Module):
    def __init__(self, inplanes, outplanes):
        super(ContourAwareDecoder, self).__init__()
        self.upsample1 = UpSampleDeconv(inplanes, inplanes, scale=4)
        self.upsample2 = UpSampleDeconv(inplanes, inplanes // 2, scale=2)
        self.final_conv_region = nn.Conv3d(in_channels=inplanes // 2, 
                                    out_channels=outplanes,
                                    kernel_size=1)
        
        self.final_conv_contour = nn.Conv3d(in_channels=inplanes // 2, 
                                    out_channels=outplanes,
                                    kernel_size=1)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_low_1, x_low_2, pyramid):
        pyramid_up = self.upsample1(pyramid)
        output = pyramid_up + x_low_2
        
        output = self.upsample2(output)
        output = output + x_low_1

        output_region = self.final_conv_region(output)
        output_region = self.sigmoid(output_region)
        
        output_contour = self.final_conv_contour(output)
        output_contour = self.sigmoid(output_contour)

        return output_region, output_contour
    
class ContourAwareDecoder(nn.Module):
    def __init__(self, inplanes, outplanes):
        super(ContourAwareDecoder, self).__init__()
        self.upsample1 = UpSampleDeconv(inplanes, inplanes, scale=4)
        self.upsample2 = UpSampleDeconv(inplanes, inplanes // 2, scale=2)
        self.final_conv_region = nn.Conv3d(in_channels=inplanes // 2, 
                                    out_channels=outplanes,
                                    kernel_size=1)
        
        self.final_conv_contour = nn.Conv3d(in_channels=inplanes // 2, 
                                    out_channels=outplanes,
                                    kernel_size=1)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_low_1, x_low_2, pyramid):
        pyramid_up = self.upsample1(pyramid)
        output = pyramid_up + x_low_2
        
        output = self.upsample2(output)
        output = output + x_low_1

        output_region = self.final_conv_region(output)
        output_region = self.sigmoid(output_region)
        
        output_contour = self.final_conv_contour(output)
        output_contour = self.sigmoid(output_contour)

        return output_region, output_contour
    
class BaseSegmentor(nn.Module):
    def __init__(self, inplanes_encoder, channels_encoder, num_classes):
        super(BaseSegmentor, self).__init__()
        self.encoder = Encoder(inplanes=inplanes_encoder, channels=channels_encoder)
        self.aspp = ASPP(inplanes=channels_encoder*8, outplanes=channels_encoder*2)
        self.decoder = DecoderDeconv(channels_encoder*2, num_classes)
    
    def forward(self, x):
        out_low_1, out_low_2, out_high = self.encoder(x)
        out_pyramid = self.aspp(out_high)
        out = self.decoder(out_low_1, out_low_2, out_pyramid)
        
        return out

class TwoBranchSegmentor(nn.Module):
    def __init__(self, inplanes_encoder, channels_encoder, num_classes):
        super(TwoBranchSegmentor, self).__init__()
        self.encoder = Encoder(inplanes=inplanes_encoder, channels=channels_encoder)
        self.aspp = ASPP(inplanes=channels_encoder*8, outplanes=channels_encoder*2)
        self.decoder1 = DecoderDeconv(channels_encoder*2, num_classes)
        self.decoder2 = DecoderInterpolation(channels_encoder*2, num_classes)
    
    def forward(self, x):
        out_low_1, out_low_2, out_high = self.encoder(x)
        out_pyramid = self.aspp(out_high)
        out1 = self.decoder1(out_low_1, out_low_2, out_pyramid)
        out2 = self.decoder2(out_low_1, out_low_2, out_pyramid)
        return out1, out2
    
class ContourAwareSegmentor(nn.Module):
    def __init__(self, inplanes_encoder, channels_encoder, num_classes):
        super(ContourAwareSegmentor, self).__init__()
        self.encoder = Encoder(inplanes=inplanes_encoder, channels=channels_encoder)
        self.aspp = ASPP(inplanes=channels_encoder*8, outplanes=channels_encoder*2)
        self.decoder = ContourAwareDecoder(channels_encoder*2, num_classes)
        
    def forward(self, x):
        out_low_1, out_low_2, out_high = self.encoder(x)
        out_pyramid = self.aspp(out_high)
        out_region, out_contour = self.decoder(out_low_1, out_low_2, out_pyramid)
        
        return out_region, out_contour
    
class LargeCascadedModel(nn.Module):
    def __init__(self, inplanes_encoder_1, channels_encoder_1, num_classes_1,
                 inplanes_encoder_2, channels_encoder_2, num_classes_2):
        super(LargeCascadedModel, self).__init__()
        self.model1 = TwoBranchSegmentor(inplanes_encoder=inplanes_encoder_1, 
                                        channels_encoder=channels_encoder_1, 
                                        num_classes=num_classes_1)
        self.model2 = ContourAwareSegmentor(inplanes_encoder=inplanes_encoder_2,
                                           channels_encoder = channels_encoder_2,
                                           num_classes=num_classes_2)
        
    def forward(self, x):
        out1, out2 = self.model1(x)
        step2_input = torch.cat([x, out1], dim=1)
        out_region, out_contour = self.model2(step2_input)
        
        return out1, out2, out_region, out_contour