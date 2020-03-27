from ImageBranchEncoder import ImageBranchEncoder
from ImageBranchDecoder import ImageBranchDecoder
from DepthBranchEncoder import DepthBranchEncoder

import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn import BatchNorm2d as bn


class NonLocalBlock(nn.Module):
    """ NonLocalBlock Module"""
    def __init__(self, in_channels):
        super(NonLocalBlock, self).__init__()

        conv_nd = nn.Conv2d

        self.in_channels = in_channels
        self.inter_channels = self.in_channels // 2

        self.ImageAfterASPP_bnRelu = nn.Sequential(
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU(inplace=True),
        )

        self.DepthAfterASPP_bnRelu = nn.Sequential(
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU(inplace=True),
        )

        self.R_g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)
        self.R_theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)
        self.R_phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)
        self.R_W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                    kernel_size=1, stride=1, padding=0)

        self.F_g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)
        self.F_theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)
        self.F_phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)
        self.F_W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                         kernel_size=1, stride=1, padding=0)

    def forward(self, self_fea, mutual_fea, alpha, selfImage):

        if selfImage:
            selfNonLocal_fea = self.ImageAfterASPP_bnRelu(self_fea)
            mutualNonLocal_fea = self.DepthAfterASPP_bnRelu(mutual_fea)

            batch_size = selfNonLocal_fea.size(0)

            g_x = self.R_g(selfNonLocal_fea).view(batch_size, self.inter_channels, -1)
            g_x = g_x.permute(0, 2, 1)

            # using mutual feature to generate attention
            theta_x = self.F_theta(mutualNonLocal_fea).view(batch_size, self.inter_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)
            phi_x = self.F_phi(mutualNonLocal_fea).view(batch_size, self.inter_channels, -1)
            f = torch.matmul(theta_x, phi_x)

            # using self feature to generate attention
            self_theta_x = self.R_theta(selfNonLocal_fea).view(batch_size, self.inter_channels, -1)
            self_theta_x = self_theta_x.permute(0, 2, 1)
            self_phi_x = self.R_phi(selfNonLocal_fea).view(batch_size, self.inter_channels, -1)
            self_f = torch.matmul(self_theta_x, self_phi_x)

            # add self_f and mutual f
            f_div_C = F.softmax(alpha*f + self_f, dim=-1)

            y = torch.matmul(f_div_C, g_x)
            y = y.permute(0, 2, 1).contiguous()
            y = y.view(batch_size, self.inter_channels, *selfNonLocal_fea.size()[2:])
            W_y = self.R_W(y)
            z = W_y + self_fea
            return z

        else:
            selfNonLocal_fea = self.DepthAfterASPP_bnRelu(self_fea)
            mutualNonLocal_fea = self.ImageAfterASPP_bnRelu(mutual_fea)

            batch_size = selfNonLocal_fea.size(0)

            g_x = self.F_g(selfNonLocal_fea).view(batch_size, self.inter_channels, -1)
            g_x = g_x.permute(0, 2, 1)

            # using mutual feature to generate attention
            theta_x = self.R_theta(mutualNonLocal_fea).view(batch_size, self.inter_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)
            phi_x = self.R_phi(mutualNonLocal_fea).view(batch_size, self.inter_channels, -1)
            f = torch.matmul(theta_x, phi_x)

            # using self feature to generate attention
            self_theta_x = self.F_theta(selfNonLocal_fea).view(batch_size, self.inter_channels, -1)
            self_theta_x = self_theta_x.permute(0, 2, 1)
            self_phi_x = self.F_phi(selfNonLocal_fea).view(batch_size, self.inter_channels, -1)
            self_f = torch.matmul(self_theta_x, self_phi_x)

            # add self_f and mutual f
            f_div_C = F.softmax(alpha*f+self_f, dim=-1)

            y = torch.matmul(f_div_C, g_x)
            y = y.permute(0, 2, 1).contiguous()
            y = y.view(batch_size, self.inter_channels, *selfNonLocal_fea.size()[2:])
            W_y = self.F_W(y)
            z = W_y + self_fea
            return z


class _DenseAsppBlock(nn.Sequential):
    """ ConvNet block for building DenseASPP. """

    def __init__(self, input_num, num1, num2, dilation_rate):
        super(_DenseAsppBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=input_num, out_channels=num1, kernel_size=1)
        self.bn1 = bn(num1, momentum=0.0003)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(in_channels=num1, out_channels=num2, kernel_size=3,
                               dilation=dilation_rate, padding=dilation_rate)
        self.bn2 = bn(num2, momentum=0.0003)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, input):

        feature = self.relu1(self.bn1(self.conv1(input)))
        feature = self.relu2(self.bn2(self.conv2(feature)))

        return feature


class DASPPmodule(nn.Module):
    def __init__(self):
        super(DASPPmodule, self).__init__()
        num_features = 512
        d_feature1 = 176
        d_feature0 = num_features//2

        self.AvgPool = nn.Sequential(
            nn.AvgPool2d([32, 32], [32, 32]),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Upsample(size=32, mode='nearest'),
        )
        self.ASPP_2 = _DenseAsppBlock(input_num=num_features, num1=d_feature0, num2=d_feature1,
                                      dilation_rate=2)

        self.ASPP_4 = _DenseAsppBlock(input_num=num_features + d_feature1 * 1, num1=d_feature0, num2=d_feature1,
                                      dilation_rate=4)

        self.ASPP_8 = _DenseAsppBlock(input_num=num_features + d_feature1 * 2, num1=d_feature0, num2=d_feature1,
                                       dilation_rate=8)

        self.afterASPP = nn.Sequential(
            nn.Conv2d(in_channels=512*2 + 176*3, out_channels=512, kernel_size=1))

    def forward(self, encoder_fea):

        imgAvgPool = self.AvgPool(encoder_fea)

        aspp2 = self.ASPP_2(encoder_fea)
        feature = torch.cat([aspp2, encoder_fea], dim=1)

        aspp4 = self.ASPP_4(feature)
        feature = torch.cat([aspp4, feature], dim=1)

        aspp8 = self.ASPP_8(feature)
        feature = torch.cat([aspp8, feature], dim=1)

        asppFea = torch.cat([feature, imgAvgPool], dim=1)
        AfterASPP = self.afterASPP(asppFea)

        return AfterASPP


class ImageDepthNet(nn.Module):
    def __init__(self, n_channels):
        super(ImageDepthNet, self).__init__()

        # encoder part
        self.ImageBranchEncoder = ImageBranchEncoder(n_channels)
        self.DepthBranchEncoder = DepthBranchEncoder(n_channels)

        self.ImageBranch_fc7_1 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        self.DepthBranch_fc7_1 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        self.affinityAttConv = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=2, kernel_size=1),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),
        )

        # DASPP
        self.ImageBranch_DASPP = DASPPmodule()
        self.DepthBranch_DASPP = DASPPmodule()

        # S2MA module
        self.NonLocal = NonLocalBlock(in_channels=512)

        self.image_bn_relu = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True))

        self.depth_bn_relu = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True))

        # decoder part
        self.ImageBranchDecoder = ImageBranchDecoder()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight),
                nn.init.constant_(m.bias, 0),

    def forward(self, image_Input, depth_Input):

        image_feas = self.ImageBranchEncoder(image_Input)
        ImageAfterDASPP = self.ImageBranch_DASPP(self.ImageBranch_fc7_1(image_feas[-1]))

        depth_feas = self.DepthBranchEncoder(depth_Input)
        DepthAfterDASPP = self.DepthBranch_DASPP(self.DepthBranch_fc7_1(depth_feas[-1]))

        bs, ch, hei, wei = ImageAfterDASPP.size()

        affinityAtt = F.softmax(self.affinityAttConv(torch.cat([ImageAfterDASPP, DepthAfterDASPP], dim=1)))
        alphaD = affinityAtt[:, 0, :, :].reshape([bs, hei * wei, 1])
        alphaR = affinityAtt[:, 1, :, :].reshape([bs, hei * wei, 1])

        alphaD = alphaD.expand([bs, hei * wei, hei * wei])
        alphaR = alphaR.expand([bs, hei * wei, hei * wei])

        ImageAfterAtt1 = self.NonLocal(ImageAfterDASPP, DepthAfterDASPP, alphaD, selfImage=True)
        DepthAfterAtt1 = self.NonLocal(DepthAfterDASPP, ImageAfterDASPP, alphaR, selfImage=False)

        ImageAfterAtt = self.image_bn_relu(ImageAfterAtt1)
        DepthAfterAtt = self.depth_bn_relu(DepthAfterAtt1)

        outputs_image, outputs_depth = self.ImageBranchDecoder(image_feas, ImageAfterAtt, depth_feas, DepthAfterAtt)
        return outputs_image, outputs_depth

    def init_parameters(self, pretrain_vgg16_1024):

        rgb_conv_blocks = [self.ImageBranchEncoder.conv1,
                       self.ImageBranchEncoder.conv2,
                       self.ImageBranchEncoder.conv3,
                       self.ImageBranchEncoder.conv4,
                       self.ImageBranchEncoder.conv5,
                       self.ImageBranchEncoder.fc6,
                       self.ImageBranchEncoder.fc7]

        depth_conv_blocks = [self.DepthBranchEncoder.conv1,
                       self.DepthBranchEncoder.conv2,
                       self.DepthBranchEncoder.conv3,
                       self.DepthBranchEncoder.conv4,
                       self.DepthBranchEncoder.conv5,
                       self.DepthBranchEncoder.fc6,
                       self.DepthBranchEncoder.fc7]

        listkey = [['conv1_1', 'conv1_2'], ['conv2_1', 'conv2_2'], ['conv3_1', 'conv3_2', 'conv3_3'],
                   ['conv4_1', 'conv4_2', 'conv4_3'], ['conv5_1', 'conv5_2', 'conv5_3'], ['fc6'], ['fc7']]

        for idx, conv_block in enumerate(rgb_conv_blocks):
            num_conv = 0
            for l2 in conv_block:
                if isinstance(l2, nn.Conv2d):
                    num_conv += 1
                    l2.weight.data = pretrain_vgg16_1024[str(listkey[idx][num_conv - 1]) + '.weight']
                    l2.bias.data = pretrain_vgg16_1024[str(listkey[idx][num_conv - 1]) + '.bias'].squeeze(0).squeeze(0).squeeze(0).squeeze(0)

        for idx, conv_block in enumerate(depth_conv_blocks):
            num_conv = 0
            for l2 in conv_block:
                if isinstance(l2, nn.Conv2d):
                    num_conv += 1
                    l2.weight.data = pretrain_vgg16_1024[str(listkey[idx][num_conv - 1]) + '.weight']
                    l2.bias.data = pretrain_vgg16_1024[str(listkey[idx][num_conv - 1]) + '.bias'].squeeze(0).squeeze(
                        0).squeeze(0).squeeze(0)
        return self
