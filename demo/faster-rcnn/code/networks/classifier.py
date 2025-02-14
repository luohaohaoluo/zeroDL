import warnings

import torch
from torch import nn
from torchvision.ops import RoIPool

warnings.filterwarnings("ignore")


class Resnet50RoIHead(nn.Module):
    def __init__(self, n_class, roi_size, spatial_scale, classifier):
        super(Resnet50RoIHead, self).__init__()
        # --------------------------------------#
        #   self.classifier输出的特征通道就是2048
        # --------------------------------------#
        self.classifier = classifier
        # --------------------------------------#
        #   对ROIPooling后的的结果进行回归预测
        # --------------------------------------#
        self.cls_loc = nn.Linear(2048, n_class * 4)
        # -----------------------------------#
        #   对ROIPooling后的的结果进行分类
        # -----------------------------------#
        self.score = nn.Linear(2048, n_class)
        # -----------------------------------#
        #   权值初始化
        # -----------------------------------#
        normal_init(self.cls_loc, 0, 0.001)
        normal_init(self.score, 0, 0.01)

        # ---------------------------------------------------------#
        #   roi_size: 14, 因为self.classifier的输入size就是14
        # ---------------------------------------------------------#
        self.roi = RoIPool((roi_size, roi_size), spatial_scale)

    def forward(self, x, rois, roi_indices, img_size):
        """

        :param x: 此前backbone输出的特征信息x
        :param rois: 筛选出来的每个样本的600个roi框
        :param roi_indices: 对应每个框属于batch中哪一个样本
        :param img_size: 输入图像原尺寸
        :return:
        """

        n, _, _, _ = x.shape
        if x.is_cuda:
            roi_indices = roi_indices.cuda()
            rois = rois.cuda()

        # -----------------------------------------------------------------#
        #  rois: (batch, 600, 4) ----> (batch * 600, 4)
        #  roi_indices: (batch, 600) - ---> (batch * 600)
        # -----------------------------------------------------------------#
        rois = torch.flatten(rois, 0, 1)
        roi_indices = torch.flatten(roi_indices, 0, 1)

        # -----------------------------------------------------------------#
        #  将ROIs的高度坐标从图像坐标系转换为特征图坐标系
        #  先除以img_size得到与原图的一个缩放比例，然后再乘以对应x的W或H
        # -----------------------------------------------------------------#
        rois_feature_map = torch.zeros_like(rois)
        rois_feature_map[:, [0, 2]] = rois[:, [0, 2]] / img_size[1] * x.size()[3]  # W
        rois_feature_map[:, [1, 3]] = rois[:, [1, 3]] / img_size[0] * x.size()[2]  # H
        # print(f"torch.unique(rois_feature_map):{torch.unique(rois_feature_map)}")
        #  indices_and_rois ---> (batch * 600, 4+1)
        indices_and_rois = torch.cat([roi_indices[:, None], rois_feature_map], dim=1)
        # print(f"indices_and_rois.shape: {indices_and_rois.shape}")
        # --------------------------------------------------#
        #   利用建议框对公用特征层进行截取
        #   x.shape: torch.Size([5, 1024, 14, 14])
        #   pool ---> (batch * 600, 1024, 14, 14)
        # --------------------------------------------------#
        pool = self.roi(x, indices_and_rois)
        # print(f"pool.shape: {pool.shape}")
        # -----------------------------------#
        #   利用classifier网络进行特征提取
        #   fc7 ---> (batch * 600, 2048, 1, 1)
        # -----------------------------------#
        fc7 = self.classifier(pool)
        # print(f"fc7.shape: {fc7.shape}")

        fc7 = fc7.view(fc7.size(0), -1)

        '''
            roi_cls_locs ----> (batch * 600, n_class*4)
            roi_scores   ----> (batch * 600, n_class)
        '''
        roi_cls_locs = self.cls_loc(fc7)
        roi_scores = self.score(fc7)
        '''
            roi_cls_locs ----> (batch, 600, n_class*4)
            roi_scores   ----> (batch, 600, n_class)
        '''
        roi_cls_locs = roi_cls_locs.view(n, -1, roi_cls_locs.size(1))
        roi_scores = roi_scores.view(n, -1, roi_scores.size(1))
        return roi_cls_locs, roi_scores


def normal_init(m, mean, stddev, truncated=False):
    if truncated:
        m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
    else:
        m.weight.data.normal_(mean, stddev)
        m.bias.data.zero_()
