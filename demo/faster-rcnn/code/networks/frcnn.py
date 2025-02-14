import torch.nn as nn
import torch

from networks.classifier import Resnet50RoIHead
from networks.resnet50 import resnet50
from networks.rpn import RegionProposalNetwork


class FasterRCNN(nn.Module):
    def __init__(self, num_classes,
                 mode="training",
                 feat_stride=16,
                 anchor_scales=[8, 16, 32],
                 ratios=[0.5, 1, 2],
                 backbone='resnet50',
                 pretrained=False):
        super(FasterRCNN, self).__init__()
        self.feat_stride = feat_stride
        # ---------------------------------#
        #   以resnet50为backbone
        # ---------------------------------#
        if backbone == 'resnet50':
            self.extractor, classifier = resnet50(pretrained)
            # ---------------------------------#
            #   构建classifier网络
            # ---------------------------------#
            self.rpn = RegionProposalNetwork(
                1024, 512,
                ratios=ratios,
                anchor_scales=anchor_scales,
                feat_stride=self.feat_stride,
                mode=mode
            )
            # ---------------------------------#
            #   构建classifier网络
            # ---------------------------------#
            self.head = Resnet50RoIHead(
                n_class=num_classes + 1,
                roi_size=14,
                spatial_scale=1,
                classifier=classifier
            )

    def forward(self, x, scale=1., mode="forward"):
        if mode == "forward":
            # ---------------------------------#
            #   计算输入图片的大小
            # ---------------------------------#
            img_size = x.shape[2:]
            # ---------------------------------#
            #   利用主干网络提取特征
            # ---------------------------------#
            base_feature = self.extractor.forward(x)
            # print(f"base_feature.shape:{base_feature.shape}")
            # ---------------------------------#
            #   获得建议框
            # ---------------------------------#
            _, _, rois, roi_indices, _ = self.rpn.forward(base_feature, img_size, scale)
            # ---------------------------------------#
            #   获得classifier的分类结果和回归结果
            # ---------------------------------------#
            roi_cls_locs, roi_scores = self.head.forward(base_feature, rois, roi_indices, img_size)
            return roi_cls_locs, roi_scores, rois, roi_indices
        elif mode == "extractor":
            # ---------------------------------#
            #   利用主干网络提取特征
            # ---------------------------------#
            base_feature = self.extractor.forward(x)
            return base_feature
        elif mode == "rpn":
            base_feature, img_size = x
            # ---------------------------------#
            #   获得建议框
            # ---------------------------------#
            rpn_locs, rpn_scores, rois, roi_indices, anchor = self.rpn.forward(base_feature, img_size, scale)
            return rpn_locs, rpn_scores, rois, roi_indices, anchor
        elif mode == "head":
            base_feature, rois, roi_indices, img_size = x
            # ---------------------------------------#
            #   获得classifier的分类结果和回归结果
            # ---------------------------------------#
            roi_cls_locs, roi_scores = self.head.forward(base_feature, rois, roi_indices, img_size)
            return roi_cls_locs, roi_scores

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()


if __name__ == "__main__":
    x = torch.rand((5, 3, 224, 224))
    # 这里的类别没有包括背景，算上背景1001类
    net = FasterRCNN(num_classes=1000, backbone='resnet50')
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


    # 打印模型参数量
    print(f"模型参数量: {count_parameters(net)}")
    y_roi_cls_locs, y_roi_scores, y_rois, y_roi_indices = net(x)

    """
        y_roi_cls_locs: 预测框对所有类别的回归预测
        y_roi_scores:   预测框对所有类别的类别预测
        y_rois:         预测框
        y_roi_indices:  预测框的索引号，同一批次的索引号相同
    """
    """
        y_roi_cls_locs.shape: torch.Size([5, 600, 4004])
        y_roi_scores.shape: torch.Size([5, 600, 1001])
        y_rois.shape: torch.Size([5, 600, 4])
        y_roi_indices.shape: torch.Size([5, 600])
    """
    print(f"y_roi_cls_locs.shape: {y_roi_cls_locs.shape}")
    print(f"y_roi_scores.shape: {y_roi_scores.shape}")
    print(f"y_rois.shape: {y_rois.shape}")
    print(f"y_roi_indices.shape: {y_roi_indices.shape}")





