import torch.nn as nn
import torch
import torch.nn.functional as F
import parameter


class decoder_module_part1(nn.Module):
    def __init__(self, in_channels, out_channels, fusing=True):
        super(decoder_module_part1, self).__init__()
        if fusing:
            self.enc_fea_proc = nn.Sequential(
                nn.BatchNorm2d(in_channels, momentum=parameter.bn_momentum),
                nn.ReLU(inplace=True),
            )
            in_channels = in_channels*2
        self.decoding1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, enc_fea, dec_fea=None):
        if dec_fea is not None:
            enc_fea = self.enc_fea_proc(enc_fea)
            if dec_fea.size(2) != enc_fea.size(2):
                dec_fea = F.upsample(dec_fea, size=[enc_fea.size(2), enc_fea.size(3)], mode='bilinear', align_corners=True)
            enc_fea = torch.cat([enc_fea, dec_fea], dim=1)
        output = self.decoding1(enc_fea)

        return output


class decoder_module_part2(nn.Module):
    def __init__(self, out_channels):
        super(decoder_module_part2, self).__init__()

        self.decoding1_resPart = nn.Sequential(
            nn.BatchNorm2d(out_channels, momentum=parameter.bn_momentum),
            nn.ReLU(inplace=True),
        )
        self.decoding2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels, momentum=parameter.bn_momentum),
            nn.ReLU(inplace=True),
        )

    def forward(self, enc_fea):

        output = self.decoding1_resPart(enc_fea)
        output = self.decoding2(output)

        return output


class DepthBranchDecoder(nn.Module):
    def __init__(self):

        super(DepthBranchDecoder, self).__init__()
        channels = [64, 128, 256, 512, 512, 512]

        self.decoder6_part1 = decoder_module_part1(channels[5], channels[4], False)
        self.decoder6_part2 = decoder_module_part2(channels[4])

        self.decoder5_part1 = decoder_module_part1(channels[4], channels[3])
        self.decoder5_part2 = decoder_module_part2(channels[3])

        self.decoder4_part1 = decoder_module_part1(channels[3], channels[2])
        self.decoder4_part2 = decoder_module_part2(channels[2])

        self.decoder3_part1 = decoder_module_part1(channels[2], channels[1])
        self.decoder3_part2 = decoder_module_part2(channels[1])

        self.decoder2_part1 = decoder_module_part1(channels[1], channels[0])
        self.decoder2_part2 = decoder_module_part2(channels[0])

        self.decoder1_part1 = decoder_module_part1(channels[0], channels[0])
        self.decoder1_part2 = decoder_module_part2(channels[0])

        self.conv_loss6 = nn.Conv2d(in_channels=channels[4], out_channels=1, kernel_size=3, padding=1)
        self.conv_loss5 = nn.Conv2d(in_channels=channels[3], out_channels=1, kernel_size=3, padding=1)
        self.conv_loss4 = nn.Conv2d(in_channels=channels[2], out_channels=1, kernel_size=3, padding=1)
        self.conv_loss3 = nn.Conv2d(in_channels=channels[1], out_channels=1, kernel_size=3, padding=1)
        self.conv_loss2 = nn.Conv2d(in_channels=channels[0], out_channels=1, kernel_size=3, padding=1)
        self.conv_loss1 = nn.Conv2d(in_channels=channels[0], out_channels=1, kernel_size=3, padding=1)

    def forward(self, enc_fea, AfterDASPP):

        encoder_conv1, encoder_conv2, encoder_conv3, encoder_conv4, encoder_conv5, x7 = enc_fea

        dec_fea_6_part1 = self.decoder6_part1(AfterDASPP)
        dec_fea_6_part2 = self.decoder6_part2(dec_fea_6_part1)

        mask6 = self.conv_loss6(dec_fea_6_part2)

        dec_fea_5_part1 = self.decoder5_part1(encoder_conv5, dec_fea_6_part2)
        dec_fea_5_part2 = self.decoder5_part2(dec_fea_5_part1)

        mask5 = self.conv_loss5(dec_fea_5_part2)

        dec_fea_4_part1 = self.decoder4_part1(encoder_conv4, dec_fea_5_part2)
        dec_fea_4_part2 = self.decoder4_part2(dec_fea_4_part1)

        mask4 = self.conv_loss4(dec_fea_4_part2)

        dec_fea_3_part1 = self.decoder3_part1(encoder_conv3, dec_fea_4_part2)
        dec_fea_3_part2 = self.decoder3_part2(dec_fea_3_part1)

        mask3 = self.conv_loss3(dec_fea_3_part2)

        dec_fea_2_part1 = self.decoder2_part1(encoder_conv2, dec_fea_3_part2)
        dec_fea_2_part2 = self.decoder2_part2(dec_fea_2_part1)

        mask2 = self.conv_loss2(dec_fea_2_part2)

        dec_fea_1_part1 = self.decoder1_part1(encoder_conv1, dec_fea_2_part2)
        dec_fea_1_part2 = self.decoder1_part2(dec_fea_1_part1)

        mask1 = self.conv_loss1(dec_fea_1_part2)

        return mask6, mask5, mask4, mask3, mask2, mask1
