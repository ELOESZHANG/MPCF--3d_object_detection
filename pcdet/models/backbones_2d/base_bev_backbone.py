import numpy as np
import torch
import torch.nn as nn



# class Second_Enhance(nn.Module):
#
#     # 输入是256td
#     def __init__(self, in_ch=256, out_ch=128):
#         super(Second_Enhance, self).__init__()
#
#         # 卷积参数设置
#         n1 = 128
#         filters = [n1, n1 * 2]
#
#         # 最大池化层
#         self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
#         # self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
#
#         # 左边特征提取卷积层
#         self.Conv1 = self.conv_block(in_ch, filters[0])
#         self.Conv2 = self.conv_block(filters[0], filters[1])
#
#
#         # 右边特征融合反卷积层
#         self.Up2 = self.up_conv(filters[1], filters[0])
#         self.Up_conv2 = self.conv_block(filters[1], filters[0])
#
#         self.Conv = self.conv_block(filters[0], out_ch)
#
#         self.conv_loop = nn.Sequential(  #128
#                         nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
#                         nn.BatchNorm2d(out_ch, eps=1e-3, momentum=0.01),
#                         nn.ReLU())
#
#     def conv_block(self, in_ch, out_ch):
#         conv = nn.Sequential(
#             nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(out_ch, eps=1e-3, momentum=0.01),
#             nn.ReLU(),
#             nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(out_ch, eps=1e-3, momentum=0.01),
#             nn.ReLU())
#         return conv
#
#     def up_conv(self, in_ch, out_ch):
#         upconv = nn.Sequential(
#             nn.Upsample(scale_factor=2),
#             nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(out_ch, eps=1e-3, momentum=0.01),
#             nn.ReLU()
#         )
#         return upconv
#
#     # 前向计算，输出一张与原图相同尺寸的图片矩阵
#     def forward(self, x):
#         e1 = self.Conv1(x)
#
#         e2 = self.Maxpool1(e1)
#         e2 = self.Conv2(e2)
#
#         d2 = self.Up2(e2)
#         d2 = torch.cat((e1, d2), dim=1)  # 将e1特征图与d1特征图横向拼接
#         d2 = self.Up_conv2(d2)
#
#         out = self.Conv(d2)
#         out = self.conv_loop(out)
#
#
#         return out
class Second_Enhance(nn.Module):

    # 输入是256td
    def __init__(self, in_ch=256, out_ch=256):
        super(Second_Enhance, self).__init__()

        # 卷积参数设置
        n1 = 128
        filters = [n1, n1 * 2, n1 * 4]

        # 左边特征提取卷积层
        self.Conv1 = self.loop_block(in_ch, filters[0]) #256, 128
        self.Conv2 = self.conv_block(filters[0], filters[1]) #128, 128
        self.Conv3 = self.conv_block(filters[1], filters[2])  # 128, 256

        # 右边特征融合反卷积层
        self.Up2 = self.up_block(filters[2], filters[1])
        self.Up1 = self.up_block(filters[1], filters[0])

        self.up_conv1 = self.loop_block(filters[1]+filters[1], filters[1])
        self.up_conv2 = self.loop_block(filters[0]+filters[0], filters[0])

        self.conv_loop1 = self.loop_block(filters[0], out_ch)

    def conv_block(self, in_ch, out_ch):
        conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_ch, eps=1e-3, momentum=0.01),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch, eps=1e-3, momentum=0.01),
            nn.ReLU())

        return conv

    def loop_block(self, in_ch, out_ch):
        conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch, eps=1e-3, momentum=0.01),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch, eps=1e-3, momentum=0.01),
            nn.ReLU())

        return conv

    def up_block(self, in_ch, out_ch):
        upconv = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_ch, eps=1e-3, momentum=0.01),
            nn.ReLU(),
            nn.ConvTranspose2d(out_ch, out_ch, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(out_ch, eps=1e-3, momentum=0.01),
            nn.ReLU())
        return upconv

    # 前向计算，输出一张与原图相同尺寸的图片矩阵
    def forward(self, x):
        e1 = self.Conv1(x) #1*256*200*176
        e2 = self.Conv2(e1)
        e3 = self.Conv3(e2)

        d2 = self.Up2(e3)
        d2 = torch.cat((d2, e2), dim=1)  # 将e1特征图与d1特征图横向拼接
        d2 = self.up_conv1(d2)

        d1 = self.Up1(d2)
        d1 = torch.cat((d1, e1), dim=1)
        d1 = self.up_conv2(d1)
        out= self.conv_loop1(d1)

        return out


class BaseBEVBackbone(nn.Module):
    def __init__(self, model_cfg, input_channels):
        super().__init__()
        self.model_cfg = model_cfg

        if self.model_cfg.get('LAYER_NUMS', None) is not None:
            assert len(self.model_cfg.LAYER_NUMS) == len(self.model_cfg.LAYER_STRIDES) == len(self.model_cfg.NUM_FILTERS)
            layer_nums = self.model_cfg.LAYER_NUMS
            layer_strides = self.model_cfg.LAYER_STRIDES
            num_filters = self.model_cfg.NUM_FILTERS
        else:
            layer_nums = layer_strides = num_filters = []

        if self.model_cfg.get('UPSAMPLE_STRIDES', None) is not None:
            assert len(self.model_cfg.UPSAMPLE_STRIDES) == len(self.model_cfg.NUM_UPSAMPLE_FILTERS)
            num_upsample_filters = self.model_cfg.NUM_UPSAMPLE_FILTERS
            upsample_strides = self.model_cfg.UPSAMPLE_STRIDES
        else:
            upsample_strides = num_upsample_filters = []

        num_levels = len(layer_nums)
        c_in_list = [input_channels, *num_filters[:-1]]
        self.blocks = nn.ModuleList()
        self.deblocks = nn.ModuleList()
        for idx in range(num_levels):
            cur_layers = [
                nn.ZeroPad2d(1),
                nn.Conv2d(
                    c_in_list[idx], num_filters[idx], kernel_size=3,
                    stride=layer_strides[idx], padding=0, bias=False
                ),
                nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                nn.ReLU()
            ]
            for k in range(layer_nums[idx]):
                cur_layers.extend([
                    nn.Conv2d(num_filters[idx], num_filters[idx], kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                    nn.ReLU()
                ])
            self.blocks.append(nn.Sequential(*cur_layers))
            if len(upsample_strides) > 0:
                stride = upsample_strides[idx]
                if stride >= 1:
                    self.deblocks.append(nn.Sequential(
                        nn.ConvTranspose2d(
                            num_filters[idx], num_upsample_filters[idx],
                            upsample_strides[idx],
                            stride=upsample_strides[idx], bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))
                else:
                    stride = np.round(1 / stride).astype(np.int)
                    self.deblocks.append(nn.Sequential(
                        nn.Conv2d(
                            num_filters[idx], num_upsample_filters[idx],
                            stride,
                            stride=stride, bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))

        c_in = sum(num_upsample_filters)
        if len(upsample_strides) > num_levels:
            self.deblocks.append(nn.Sequential(
                nn.ConvTranspose2d(c_in, c_in, upsample_strides[-1], stride=upsample_strides[-1], bias=False),
                nn.BatchNorm2d(c_in, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ))

        self.num_bev_features = c_in
        # self.conv_out=nn.Sequential(
        #                 nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
        #                 nn.BatchNorm2d(256, eps=1e-3, momentum=0.01),
        #                 nn.ReLU())
        # self.second = Second_Enhance(256, 256)

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                spatial_features
        Returns:
        """
        spatial_features = data_dict['spatial_features']
        ups = []
        ret_dict = {}

        x = spatial_features

        # [1,256,200,176]

        # x = self.second(x)
        # x = self.conv_out(x)

        for i in range(len(self.blocks)):
            x = self.blocks[i](x)

            stride = int(spatial_features.shape[2] / x.shape[2])
            ret_dict['spatial_features_%dx' % stride] = x
            if len(self.deblocks) > 0:
                ups.append(self.deblocks[i](x))
            else:
                ups.append(x)

        if len(ups) > 1:
            x = torch.cat(ups, dim=1)
        elif len(ups) == 1:
            x = ups[0]

        if len(self.deblocks) > len(self.blocks):
            x = self.deblocks[-1](x)
        # [1,256,200,176]

        data_dict['spatial_features_2d'] = x

        return data_dict
