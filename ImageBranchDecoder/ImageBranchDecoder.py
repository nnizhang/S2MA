import torch.nn as nn
import torch
import torch.nn.functional as F
import parameter
from DepthBranchDecoder import DepthBranchDecoder


class Res_part(nn.Module):

    def __init__(self, in_channels):
        super(Res_part, self).__init__()
        self.res_bn_relu = nn.Sequential(
            nn.BatchNorm2d(in_channels*2),
            nn.ReLU(inplace=True),
        )
        self.res_conv = nn.Conv2d(in_channels*2, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, ImageFea, DepthFea):

        ImageFlow_Fea = torch.cat([ImageFea, DepthFea], dim=1)
        ImageFlow_resFea = self.res_bn_relu(ImageFlow_Fea)
        ImageFlow_resFea = self.res_conv(ImageFlow_resFea)

        return ImageFea + ImageFlow_resFea


class decoder_module(nn.Module):
    def __init__(self, in_channels, out_channels, fusing=True):
        super(decoder_module, self).__init__()
        if fusing:
            self.enc_fea_proc = nn.Sequential(
                nn.BatchNorm2d(in_channels, momentum=parameter.bn_momentum),
                nn.ReLU(inplace=True),
            )
            in_channels = in_channels*2

            self.ResPart = Res_part(out_channels)

        self.decoding1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels, momentum=parameter.bn_momentum),
            nn.ReLU(inplace=True),
        )

        self.decoding2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels, momentum=parameter.bn_momentum),
            nn.ReLU(inplace=True),
        )

    def forward(self, enc_fea, depth_fea=None, dec_fea=None):
        if (dec_fea is not None) and (depth_fea is not None):
            # process encoder feature
            enc_fea = self.enc_fea_proc(enc_fea)
            if dec_fea.size(2) != enc_fea.size(2):
                dec_fea = F.upsample(dec_fea, size=[enc_fea.size(2), enc_fea.size(3)], mode='bilinear', align_corners=True)
            enc_fea = torch.cat([enc_fea, dec_fea], dim=1)

            # split conv1/bn/relu to conv1, ResPart, bn, relu
            # conv1
            output = self.decoding1[0](enc_fea)

            output = self.ResPart(output, depth_fea)

            # bn/relu
            output = self.decoding1[1](output)
            output = self.decoding1[2](output)

            # conv2
            output = self.decoding2(output)
        else:
            output = self.decoding1(enc_fea)
            output = self.decoding2(output)

        return output


class ImageBranchDecoder(nn.Module):
    def __init__(self):

        super(ImageBranchDecoder, self).__init__()
        channels = [64, 128, 256, 512, 512, 512]

        self.decoder6 = decoder_module(channels[5], channels[4], False)
        self.decoder5 = decoder_module(channels[4], channels[3])
        self.decoder4 = decoder_module(channels[3], channels[2])
        self.decoder3 = decoder_module(channels[2], channels[1])
        self.decoder2 = decoder_module(channels[1], channels[0])
        self.decoder1 = decoder_module(channels[0], channels[0])

        self.conv_loss6 = nn.Conv2d(in_channels=channels[4], out_channels=1, kernel_size=3, padding=1)
        self.conv_loss5 = nn.Conv2d(in_channels=channels[3], out_channels=1, kernel_size=3, padding=1)
        self.conv_loss4 = nn.Conv2d(in_channels=channels[2], out_channels=1, kernel_size=3, padding=1)
        self.conv_loss3 = nn.Conv2d(in_channels=channels[1], out_channels=1, kernel_size=3, padding=1)
        self.conv_loss2 = nn.Conv2d(in_channels=channels[0], out_channels=1, kernel_size=3, padding=1)
        self.conv_loss1 = nn.Conv2d(in_channels=channels[0], out_channels=1, kernel_size=3, padding=1)

        self.DepthBranchDecoder = DepthBranchDecoder()

    def forward(self, image_feas, ImageAfterAtt, depth_feas, DepthAfterAtt):

        encoder_conv1, encoder_conv2, encoder_conv3, encoder_conv4, encoder_conv5, x7 = image_feas
        depth_encoder_conv1, depth_encoder_conv2, depth_encoder_conv3, depth_encoder_conv4, depth_encoder_conv5, depth_x7 = depth_feas

        # depth (decoder6)
        depth_dec_fea_6_part1 = self.DepthBranchDecoder.decoder6_part1(DepthAfterAtt)
        depth_dec_fea_6_part2 = self.DepthBranchDecoder.decoder6_part2(depth_dec_fea_6_part1)
        depth_mask6 = self.DepthBranchDecoder.conv_loss6(depth_dec_fea_6_part2)
        # image (decoder6)
        dec_fea_6 = self.decoder6(ImageAfterAtt)
        mask6 = self.conv_loss6(dec_fea_6)

        # depth (decoder5)
        depth_dec_fea_5_part1 = self.DepthBranchDecoder.decoder5_part1(depth_encoder_conv5, depth_dec_fea_6_part2)
        depth_dec_fea_5_part2 = self.DepthBranchDecoder.decoder5_part2(depth_dec_fea_5_part1)
        depth_mask5 = self.DepthBranchDecoder.conv_loss5(depth_dec_fea_5_part2)
        # image (decoder5)
        dec_fea_5 = self.decoder5(encoder_conv5, depth_dec_fea_5_part1, dec_fea_6)
        mask5 = self.conv_loss5(dec_fea_5)

        # depth (decoder4)
        depth_dec_fea_4_part1 = self.DepthBranchDecoder.decoder4_part1(depth_encoder_conv4, depth_dec_fea_5_part2)
        depth_dec_fea_4_part2 = self.DepthBranchDecoder.decoder4_part2(depth_dec_fea_4_part1)
        depth_mask4 = self.DepthBranchDecoder.conv_loss4(depth_dec_fea_4_part2)
        # image (decoder4)
        dec_fea_4 = self.decoder4(encoder_conv4, depth_dec_fea_4_part1, dec_fea_5)
        mask4 = self.conv_loss4(dec_fea_4)

        # depth (decoder3)
        depth_dec_fea_3_part1 = self.DepthBranchDecoder.decoder3_part1(depth_encoder_conv3, depth_dec_fea_4_part2)
        depth_dec_fea_3_part2 = self.DepthBranchDecoder.decoder3_part2(depth_dec_fea_3_part1)
        depth_mask3 = self.DepthBranchDecoder.conv_loss3(depth_dec_fea_3_part2)
        # image (decoder3)
        dec_fea_3 = self.decoder3(encoder_conv3, depth_dec_fea_3_part1, dec_fea_4)
        mask3 = self.conv_loss3(dec_fea_3)

        # depth (decoder2)
        depth_dec_fea_2_part1 = self.DepthBranchDecoder.decoder2_part1(depth_encoder_conv2, depth_dec_fea_3_part2)
        depth_dec_fea_2_part2 = self.DepthBranchDecoder.decoder2_part2(depth_dec_fea_2_part1)
        depth_mask2 = self.DepthBranchDecoder.conv_loss2(depth_dec_fea_2_part2)
        # image (decoder2)
        dec_fea_2 = self.decoder2(encoder_conv2, depth_dec_fea_2_part1, dec_fea_3)
        mask2 = self.conv_loss2(dec_fea_2)

        # depth (decoder1)
        depth_dec_fea_1_part1 = self.DepthBranchDecoder.decoder1_part1(depth_encoder_conv1, depth_dec_fea_2_part2)
        depth_dec_fea_1_part2 = self.DepthBranchDecoder.decoder1_part2(depth_dec_fea_1_part1)
        depth_mask1 = self.DepthBranchDecoder.conv_loss1(depth_dec_fea_1_part2)
        # image (decoder1)
        dec_fea_1 = self.decoder1(encoder_conv1, depth_dec_fea_1_part1, dec_fea_2)
        mask1 = self.conv_loss1(dec_fea_1)

        return [mask6, mask5, mask4, mask3, mask2, mask1], [depth_mask6, depth_mask5, depth_mask4, depth_mask3, depth_mask2, depth_mask1]
