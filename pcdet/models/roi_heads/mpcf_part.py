import torch
import torch.nn as nn
import torch.nn.functional as F
from pcdet.models.backbones_3d.pointnet2_backbone import PointNet2Backbone, PointNet2MSG
from torch.nn.functional import normalize

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



    def forward(self, color_point_fea, color_point_link):
        if color_point_fea.shape[0] == 0:
            return color_point_fea
        # color_point_fea [ **,9]
        # color_point_link [ **,90]

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

        return color_conv_fea

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


    def forward(self, pseudo_feas0, valid_feas0):
        B,N,_ = pseudo_feas0.size()
        dn = N
        Ra=1

        # pseudo_feas0 = normalize(pseudo_feas0, dim=-1)
        # valid_feas0  = normalize(valid_feas0, dim=-1)
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
        pseudo_feas_end = normalize(pseudo_feas_end, dim=-1)*0.1 + pseudo_feas0*(1.1-0.1*Ra)

        valid_feas_end = torch.bmm(val_Q, pse_K.transpose(-2, -1)) / dn
        # valid_feas_end = F.relu(valid_feas_end)
        valid_feas_end = torch.bmm(valid_feas_end, val_V)
        valid_feas_end = self.fc1(valid_feas_end).transpose(1, 2)
        valid_feas_end = normalize(valid_feas_end, dim=-1)*0.1 + valid_feas0*(1.1-0.1*Ra)
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
        Rb=1
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

        pse_feas_end = self.conv1(pse_feas * pse_feas_1*(1.1-0.1*Rb))  # [100,1,216]
        val_feas_end = self.conv2(val_feas * val_feas_1*(1.1-0.1*Rb))

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


class Fusion3(nn.Module):
    def __init__(self, pseudo_in, valid_in, outplanes):
        super(Fusion3, self).__init__()

        self.attention0 = TransAttention(channels=pseudo_in)
        self.attention1 = ROIAttention(channels=pseudo_in)
        self.attention2 = ROIAttention(channels=pseudo_in)

        self.conv1 = torch.nn.Conv1d(valid_in * 2, outplanes, 1)  #128+128,256,1
        self.bn1 = torch.nn.BatchNorm1d(outplanes)
        self.relu = nn.ReLU()

    def forward(self, valid_features, pseudo_features):

        pseudo_features, valid_features = self.attention0(pseudo_features, valid_features)
        pseudo_features, valid_features = self.attention1(pseudo_features, valid_features)
        pseudo_features, valid_features = self.attention2(pseudo_features, valid_features)

        # fusion_features = torch.cat([valid_features2, valid_features, pseudo_features2, pseudo_features], dim=1)
        fusion_features = torch.cat([valid_features, pseudo_features], dim=1)
        fusion_features = self.relu(self.bn1(self.conv1(fusion_features)))

        return fusion_features


