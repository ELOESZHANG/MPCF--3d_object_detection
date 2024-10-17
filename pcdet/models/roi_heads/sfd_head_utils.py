import torch
import torch.nn as nn
import torch.nn.functional as F
from pcdet.models.backbones_3d.pointnet2_backbone import PointNet2Backbone, PointNet2MSG
from torch.nn.functional import normalize

class PointNet(nn.Module):
    def __init__(self, in_channel=9, out_channels=32):
        super(PointNet, self).__init__()
        self.conv1 = torch.nn.Conv1d(in_channel, out_channels, 1)
        self.conv2 = torch.nn.Conv1d(out_channels, out_channels, 1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.out_channels = out_channels

    def forward(self, x):
        x = x.transpose(1,2)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = x.transpose(1,2)

        return x

class CPConvs(nn.Module):
    def __init__(self):
        super(CPConvs, self).__init__()
        self.pointnet1_fea = PointNet(  6,12)
        self.pointnet1_wgt = PointNet(  6,12)
        self.pointnet1_fus = PointNet(108,12)

        self.pointnet2_fea = PointNet( 12,24)
        self.pointnet2_wgt = PointNet(  6,24)
        self.pointnet2_fus = PointNet(216,24)

        self.pointnet3_fea = PointNet( 24,48)
        self.pointnet3_wgt = PointNet(  6,48)
        self.pointnet3_fus = PointNet(432,48)

    def forward(self, points_features, points_neighbor):
        if points_features.shape[0] == 0:
            return points_features

        N, F = points_features.shape
        N, M = points_neighbor.shape
        point_empty = (points_neighbor == 0).nonzero()
        points_neighbor[point_empty[:,0], point_empty[:,1]] = point_empty[:,0]

        pointnet_in_xiyiziuiviri = torch.index_select(points_features[:,[0,1,2,6,7,8]],0,points_neighbor.view(-1)).view(N,M,-1)
        pointnet_in_x0y0z0u0v0r0 = points_features[:,[0,1,2,6,7,8]].unsqueeze(dim=1).repeat([1,M,1])
        pointnet_in_xyzuvr       = pointnet_in_xiyiziuiviri - pointnet_in_x0y0z0u0v0r0
        points_features[:, 3:6] /= 255.0
        from torch.nn.functional import normalize

        points_features[:, :3] = normalize(points_features[:, :3], dim=0)
        points_features[:, 6:] = normalize(points_features[:, 6:], dim=0)

        pointnet1_in_fea        = points_features[:,:6].view(N,1,-1)
        pointnet1_out_fea       = self.pointnet1_fea(pointnet1_in_fea).view(N,-1)
        pointnet1_out_fea       = torch.index_select(pointnet1_out_fea,0,points_neighbor.view(-1)).view(N,M,-1)
        pointnet1_out_wgt       = self.pointnet1_wgt(pointnet_in_xyzuvr)
        pointnet1_feas          = pointnet1_out_fea * pointnet1_out_wgt
        pointnet1_feas          = self.pointnet1_fus(pointnet1_feas.reshape(N,1,-1)).view(N,-1)

        pointnet2_in_fea        = pointnet1_feas.view(N,1,-1)
        pointnet2_out_fea       = self.pointnet2_fea(pointnet2_in_fea).view(N,-1)
        pointnet2_out_fea       = torch.index_select(pointnet2_out_fea,0,points_neighbor.view(-1)).view(N,M,-1)
        pointnet2_out_wgt       = self.pointnet2_wgt(pointnet_in_xyzuvr)
        pointnet2_feas           = pointnet2_out_fea * pointnet2_out_wgt
        pointnet2_feas          = self.pointnet2_fus(pointnet2_feas.reshape(N,1,-1)).view(N,-1)

        pointnet3_in_fea        = pointnet2_feas.view(N,1,-1)
        pointnet3_out_fea       = self.pointnet3_fea(pointnet3_in_fea).view(N,-1)
        pointnet3_out_fea       = torch.index_select(pointnet3_out_fea,0,points_neighbor.view(-1)).view(N,M,-1)
        pointnet3_out_wgt       = self.pointnet3_wgt(pointnet_in_xyzuvr)
        pointnet3_feas           = pointnet3_out_fea * pointnet3_out_wgt
        pointnet3_feas          = self.pointnet3_fus(pointnet3_feas.reshape(N,1,-1)).view(N,-1)

        pointnet_feas     = torch.cat([pointnet3_feas, pointnet2_feas, pointnet1_feas, points_features[:,:6]], dim=-1)
        return pointnet_feas

class Attention(nn.Module):
    def __init__(self, channels):
        super(Attention, self).__init__()
        self.pseudo_in, self.valid_in = channels
        middle = self.valid_in // 4
        self.fc1 = nn.Linear(self.pseudo_in, middle)
        self.fc2 = nn.Linear(self.valid_in, middle)
        self.fc3 = nn.Linear(2*middle, 2)
        self.conv1 = nn.Sequential(nn.Conv1d(self.pseudo_in, self.valid_in, 1),
                                    nn.BatchNorm1d(self.valid_in),
                                    nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv1d(self.valid_in, self.valid_in, 1),
                                    nn.BatchNorm1d(self.valid_in),
                                    nn.ReLU())

    def forward(self, pseudo_feas, valid_feas):
        batch = pseudo_feas.size(0)
        # print('pseudo_feas',pseudo_feas.shape)
        pseudo_feas_f = pseudo_feas.transpose(1,2).contiguous().view(-1, self.pseudo_in)
        valid_feas_f = valid_feas.transpose(1,2).contiguous().view(-1, self.valid_in)
        # print('pseudo_feas_f', pseudo_feas_f.shape)

        pseudo_feas_f_ = self.fc1(pseudo_feas_f)
        valid_feas_f_ = self.fc2(valid_feas_f)
        # print('pseudo_feas_f_', pseudo_feas_f_.shape)
        pseudo_valid_feas_f = torch.cat([pseudo_feas_f_, valid_feas_f_],dim=-1)
        # print('pseudo_valid_feas_f', pseudo_valid_feas_f.shape)
        weight = torch.sigmoid(self.fc3(pseudo_valid_feas_f))
        # print('weight', weight.shape)

        pseudo_weight = weight[:,0].squeeze()
        # print('pseudo_weight', pseudo_weight.shape)
        pseudo_weight = pseudo_weight.view(batch, 1, -1)
        # print('pseudo_weight', pseudo_weight.shape)

        valid_weight = weight[:,1].squeeze()
        valid_weight = valid_weight.view(batch, 1, -1)

        pseudo_features_att = self.conv1(pseudo_feas)  * pseudo_weight
        valid_features_att     =  self.conv2(valid_feas)      *  valid_weight
        # print('pseudo_features_att', pseudo_features_att.shape)

        return pseudo_features_att, valid_features_att


class ColorEh(nn.Module):
    def color_fc(self, in_channel=9, out_channels=32):
        self.fc1 = nn.Linear(in_channel, out_channels)
        self.fc2 = nn.Linear(out_channels, out_channels)
        self.fc3 = nn.Linear(out_channels, out_channels)
        # self.dp1 = nn.Dropout(p=0.05)
        # self.dp2 = nn.Dropout(p=0.05)
        self.relu1 = nn.ReLU()
        # self.relu2 = nn.ReLU()

        FC = nn.Sequential(
            self.fc1,
            self.fc2,
            self.fc3,
            self.relu1
        )
        return FC

    def __init__(self):
        super(ColorEh, self).__init__()
        self.color_fc11 = self.color_fc(6, 18)
        self.color_fc21 = self.color_fc(18, 54)
        self.color_fc31 = self.color_fc(54, 18)
        self.color_fc41 = self.color_fc(18, 6)

        self.color_fc22 = self.color_fc(6, 54)
        self.color_fc23 = self.color_fc(486, 54)
        # self.color_fc_end = self.color_fc(102, 9)


    def forward(self, color_point_fea, color_point_link):
        if color_point_fea.shape[0] == 0:
            return color_point_fea
        # color_point_fea [ **,9]
        # color_point_link [ **,90]
        # color_point_fea=color_point_fea11.clone()
        # color_point_link=color_point_link11.clone()

        N, M = color_point_link.shape
        point_empty = (color_point_link == 0).nonzero()  # select no zero
        color_point_link[point_empty[:, 0], point_empty[:, 1]] = point_empty[:, 0]
        color_point_link = color_point_link.view(-1)

        ninei = torch.index_select(color_point_fea, 0, color_point_link)
        ninei = ninei.view(N, M, -1)
        nine0 = color_point_fea.unsqueeze(dim=-2).repeat([1, M, 1])
        ninei = ninei - nine0

        color_point_fea[:, 3:6] /= 255.0
        color_point_fea[:, :3] = normalize(color_point_fea[:, :3], dim=0)
        color_point_fea[:, 6:] = normalize(color_point_fea[:, 6:], dim=0)

        ninei = ninei[:, :, [0, 1, 2, 6, 7, 8]]

        fea1 = self.color_fc11(color_point_fea[:, :6])
        fea2 = self.color_fc21(fea1)
        fea3 = self.color_fc31(fea2)
        fea4 = self.color_fc41(fea3)

        fea2_1 = torch.index_select(fea2, 0, color_point_link).view(N, M, -1)
        fea2_1 = fea2_1 * self.color_fc22(ninei)
        fea2_1 = self.color_fc23(fea2_1.view(N, -1))

        color_conv_fea = torch.cat([fea4, fea3, fea2_1, fea1, color_point_fea[:, :6]], dim=-1) #[50001,102]
        # color_CC = self.color_fc_end(color_conv_fea)

        return color_conv_fea


class ColorCC(nn.Module):

    def colorchan_fc(self, in_channel=9, out_channels=32):
        self.fc1 = nn.Linear(in_channel, out_channels)
        self.fc2 = nn.Linear(out_channels, out_channels)
        self.fc3 = nn.Linear(out_channels, out_channels)
        # self.dp1 = nn.Dropout(p=0.05)
        # self.dp2 = nn.Dropout(p=0.05)
        self.relu1 = nn.ReLU()
        # self.relu2 = nn.ReLU()

        FC = nn.Sequential(
            self.fc1,
            self.fc2,
            self.fc3,
            self.relu1
        )
        return FC

    def __init__(self):
        super(ColorCC, self).__init__()
        self.color_fc11 = self.colorchan_fc(9, 18)
        self.color_fc21 = self.colorchan_fc(18, 54)
        self.color_fc31 = self.colorchan_fc(54, 18)
        self.color_fc41 = self.colorchan_fc(18, 6)

        self.color_fc22 = self.colorchan_fc(9, 54)
        self.color_fc23 = self.colorchan_fc(486, 54)
        self.color_fc_end = self.colorchan_fc(102, 51)

    def forward(self, color_point_fea, color_point_link, color_features_expand):
        N, M = color_point_fea.shape
        ninei = torch.index_select(color_point_fea, 0, color_point_link)
        ninei = ninei.view(N, M, -1)
        nine0 = color_point_fea.unsqueeze(dim=-2).repeat([1, M, 1])
        ninei = ninei - nine0

        fea1 = self.color_fc11(color_point_fea)
        fea2 = self.color_fc21(fea1)
        fea3 = self.color_fc31(fea2)
        fea4 = self.color_fc41(fea3)

        fea2_1 = torch.index_select(fea2, 0, color_point_link).view(N, M, -1)
        fea2_1 = fea2_1 * self.color_fc22(ninei)
        fea2_1 = self.color_fc23(fea2_1.view(N, -1))

        color_conv_fea = torch.cat([fea4, fea3, fea2_1, fea1, color_point_fea[:, :6]], dim=-1)
        color_conv_fea = self.color_fc_end(color_conv_fea)
        color_features_expand = self.color_fc_end(color_features_expand)
        color_features_expand = torch.cat([color_conv_fea, color_features_expand], dim=-1)

        return color_features_expand

class TransAttention(nn.Module):
    def __init__(self, channels):
        super(TransAttention, self).__init__()
        self.channels = channels

        self.fc1 = nn.Sequential(nn.Linear(channels, channels),
                                 nn.Linear(channels, channels),
                                 nn.Linear(channels, channels),
                                 nn.Linear(channels, channels),
                                 nn.SELU(),
                                 nn.Dropout(p=0.1, inplace=False),
                                 nn.Linear(channels, channels),
                                 nn.Linear(channels, channels),
                                 nn.Linear(channels, channels),
                                 )
        # self.fc2 = nn.Sequential(nn.Linear(channels, channels),
        #                          nn.Linear(channels, channels),
        #                          nn.Linear(channels, channels),
        #                          nn.Linear(channels, channels),
        #                          )
        # self.fc3 = nn.Sequential(nn.Linear(channels, channels),
        #                          nn.Linear(channels, channels),
        #                          nn.Linear(channels, channels),
        #                          nn.Linear(channels, channels),
        #                          nn.Linear(channels, channels),
        #                          )
        # self.fc4 = nn.Sequential(nn.Linear(channels, channels),
        #                          nn.Linear(channels, channels),
        #                          nn.ReLU()
        #                          )
        # self.conv1 = nn.Sequential(nn.Conv1d(self.channels, self.channels, 1),
        #                            nn.ReLU())
        # self.conv = nn.Sequential(nn.Conv1d(self.channels, self.channels, 1),
        #                            nn.BatchNorm1d(self.channels),
        #                            nn.ReLU())

    def forward(self, pseudo_feas0, valid_feas0):
        B,N,_ = pseudo_feas0.size()
        dn = N

        pseudo_feas = pseudo_feas0.transpose(1, 2)
        valid_feas = valid_feas0.transpose(1, 2)

        pse_Q = self.fc1(pseudo_feas)
        pse_K = self.fc1(pseudo_feas)
        pse_V = pseudo_feas
        pse_Q = F.softmax(pse_Q, dim=-2)
        pse_K = F.softmax(pse_K, dim=-1)

        val_Q = self.fc1(valid_feas)
        val_K = self.fc1(valid_feas)
        val_V = valid_feas
        val_Q = F.softmax(val_Q, dim=-2)
        val_K = F.softmax(val_K, dim=-1)

        pseudo_feas_end = torch.bmm(pse_Q, val_K.transpose(-2, -1)) / dn
        # pseudo_feas_end = F.relu(pseudo_feas_end)
        pseudo_feas_end = torch.bmm(pseudo_feas_end, pse_V)
        pseudo_feas_end = self.fc1(pseudo_feas_end).transpose(1, 2)
        pseudo_feas_end = normalize(pseudo_feas_end, dim=-1)*0.5 + pseudo_feas0

        valid_feas_end = torch.bmm(val_Q, pse_K.transpose(-2, -1)) / dn
        # valid_feas_end = F.relu(valid_feas_end)
        valid_feas_end = torch.bmm(valid_feas_end, val_V)
        valid_feas_end = self.fc1(valid_feas_end).transpose(1, 2)
        valid_feas_end = normalize(valid_feas_end, dim=-1)*0.5 + valid_feas0
        # pseudo_feas_end = self.conv(pseudo_feas_end.transpose(1, 2))
        # valid_feas_end = self.conv(valid_feas_end.transpose(1, 2))
        # print('pseudo_features_att', pseudo_features_att.shape)

        return pseudo_feas_end, valid_feas_end


class ROIAttention(nn.Module):
    def __init__(self, channels):
        super(ROIAttention, self).__init__()
        self.channels = channels

        self.fc1 = nn.Linear(self.channels * 2, self.channels * 4)
        self.fc2 = nn.Linear(self.channels * 4, self.channels * 2)
        self.fc3 = nn.Linear(self.channels * 2, self.channels)

        self.fc4p = nn.Linear(self.channels//2, self.channels//4)
        self.fc4v = nn.Linear(self.channels//2, self.channels//4)
        self.fc5p = nn.Linear(self.channels//4, 1)
        self.fc5v = nn.Linear(self.channels//4, 1)

        self.conv1 = nn.Sequential(nn.Conv1d(self.channels, self.channels, 1),
                                    nn.BatchNorm1d(self.channels),
                                    nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv1d(self.channels, self.channels, 1),
                                    nn.BatchNorm1d(self.channels),
                                    nn.ReLU())

    def forward(self, pse_feas, val_feas):
        # print('pseudo_feas',pseudo_feas.shape)       #[100, 128, 216])
        B, N, _  = pse_feas.size()
        pse_feas_1 = pse_feas.transpose(1,2).reshape(-1,N)       #[100,216,128]
        val_feas_1 = val_feas.transpose(1,2).reshape(-1,N)

        fusion_fea = torch.cat([pse_feas_1, val_feas_1], dim=-1)  #[100,216,256]
        fusion_fea = self.fc1(fusion_fea)
        fusion_fea = self.fc2(fusion_fea)
        fusion_fea = self.fc3(fusion_fea)
        C = self.channels//2
        pse_feas_1 = fusion_fea[:, :C]
        val_feas_1 = fusion_fea[:, C:]

        pse_feas_1 = self.fc4p(pse_feas_1)
        val_feas_1 = self.fc4v(val_feas_1)
        pse_feas_1 = self.fc5p(pse_feas_1)
        val_feas_1 = self.fc5v(val_feas_1)

        pse_feas_1 = torch.sigmoid(pse_feas_1).view(B, -1, 1).transpose(1, 2)
        val_feas_1 = torch.sigmoid(val_feas_1).view(B, -1, 1).transpose(1, 2)

        pse_feas_end = self.conv1(pse_feas * pse_feas_1)  # [100,1,216]
        val_feas_end = self.conv2(val_feas * val_feas_1)

        # fusion_fea = self.fc1(fusion_fea.view(-1, 2*N))#[100,216,2]
        #
        # fusion_fea = torch.sigmoid(fusion_fea).view(B,-1,2).transpose(1, 2)
        # wei_pse = fusion_fea[:, 0, :].unsqueeze(1) #[100,1,216]
        # wei_val = fusion_fea[:, 1, :].unsqueeze(1)
        # pse_feas_end = self.conv1(pse_feas * wei_pse)   #[100,1,216]
        # val_feas_end = self.conv2(val_feas * wei_val)

        return pse_feas_end, val_feas_end


class Baseline_color(nn.Module):
    def __init__(self):
        super(Baseline_color, self).__init__()

    def forward(self, points_features, points_neighbor):
        if points_features.shape[0] == 0:
            return points_features
        #points_features [ **,9]
        #points_neighbor [ **,9]

        points_features[:, 3:6] /= 255.0
        points_features[:, :3] = normalize(points_features[:, :3], dim=0)
        points_features[:, 6:] = normalize(points_features[:, 6:], dim=0)

        N, _ = points_neighbor.shape
        point_empty = (points_neighbor == 0).nonzero()  #select no zero
        points_neighbor[point_empty[:, 0], point_empty[:, 1]] = point_empty[:, 0]
        points_neighbor=points_neighbor.view(-1)

        xyz_aaa = torch.index_select(points_features, 0, points_neighbor).view(N,-1)

        pointnet_feas = torch.cat([xyz_aaa, points_features], dim=-1)
        # points_features [ **,90]
        return pointnet_feas


class GAF(nn.Module):
    def __init__(self, pseudo_in, valid_in, outplanes):
        super(GAF, self).__init__()
        # self.attention0 = Attention(channels = [pseudo_in, valid_in])
        # self.attention0 = ROIAttention(channels=pseudo_in)
        self.attention0 = TransAttention(channels=pseudo_in)
        self.attention1 = ROIAttention(channels=pseudo_in)
        self.attention2 = ROIAttention(channels=pseudo_in)
        # self.attention3 = ROIAttention(channels=pseudo_in)
        self.conv1 = torch.nn.Conv1d(valid_in * 2, outplanes, 1)  #128+128,256,1
        self.bn1 = torch.nn.BatchNorm1d(outplanes)
        self.relu = nn.ReLU()
        # self.cam = channel_attention(128, ratio=4)
        # self.se = SE(in_channel=128, ratio=4)
        # self.sam = spatial_attention(kernel_size=7)
        # self.aspp = ASPP(in_channel=128, depth=128)
        # self.row = RowAttention(128,128)
        # self.col = ColAttention(128,128)
        # self.dwconv = DWConv(128,128)
        # self.conv2 = torch.nn.Conv1d(512, 256, 1)
        # self.bn2 = torch.nn.BatchNorm1d(128)
        # self.conv3 = torch.nn.Conv1d(256, 128, 1)
        # self.conv4 = torch.nn.Conv1d(128, 128, 1)
        # self.conv5 = torch.nn.Conv1d(128, 256, 1)

    def forward(self, pseudo_features, valid_features):

        pseudo_features, valid_features = self.attention0(pseudo_features, valid_features) #128*128*216
        pseudo_features, valid_features = self.attention1(pseudo_features, valid_features)
        pseudo_features, valid_features = self.attention2(pseudo_features, valid_features)

        # fusion_features = torch.cat([valid_features2, valid_features, pseudo_features2, pseudo_features], dim=1)              #128*256*216
        fusion_features = torch.cat([valid_features, pseudo_features], dim=1)
        fusion_features = self.relu(self.bn1(self.conv1(fusion_features)))                      # 128*256*216


        return fusion_features


