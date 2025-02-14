import numpy as np
import torch
from torch.nn import functional as F
from torchvision.ops import nms


def loc2bbox(src_bbox, loc):
    if src_bbox.size()[0] == 0:
        return torch.zeros((0, 4), dtype=loc.dtype)

    src_width = torch.unsqueeze(src_bbox[:, 2] - src_bbox[:, 0], -1)
    src_height = torch.unsqueeze(src_bbox[:, 3] - src_bbox[:, 1], -1)

    # ------------------------------------------#
    #  src_ctr_x: 左上点的x坐标
    #  src_ctr_y: 左上点的y坐标
    # ------------------------------------------#
    src_ctr_x = torch.unsqueeze(src_bbox[:, 0], -1) + 0.5 * src_width
    src_ctr_y = torch.unsqueeze(src_bbox[:, 1], -1) + 0.5 * src_height

    # ------------------------------------------#
    #  [:, 0::4]: 从第0列开始，步长为4的取数据
    #  loc.shape      ----> (12996, 4)
    #  dx, dy, dw, dh ----> (12996, 1)
    # ------------------------------------------#
    dx = loc[:, 0::4]
    dy = loc[:, 1::4]
    dw = loc[:, 2::4]
    dh = loc[:, 3::4]

    ctr_x = dx * src_width + src_ctr_x
    ctr_y = dy * src_height + src_ctr_y
    w = torch.exp(dw) * src_width
    h = torch.exp(dh) * src_height

    # 格式转变, 转变成左上角右下角格式, 返回带有预测性质的先验框
    dst_bbox = torch.zeros_like(loc)
    dst_bbox[:, 0::4] = ctr_x - 0.5 * w
    dst_bbox[:, 1::4] = ctr_y - 0.5 * h
    dst_bbox[:, 2::4] = ctr_x + 0.5 * w
    dst_bbox[:, 3::4] = ctr_y + 0.5 * h

    return dst_bbox


class DecodeBox(object):
    def __init__(self, std, num_classes):
        self.std = std
        self.num_classes = num_classes + 1

    # ----------------------------------------------------#
    #   将框的中心点坐标和宽高转换为左上角和右下角坐标
    #   将框的坐标从输入图像的尺度调整到原始图像的尺度
    # ----------------------------------------------------#
    def frcnn_correct_boxes(self, box_xy, box_wh, input_shape, image_shape):
        # -----------------------------------------------------------------#
        #   把y轴放前面是因为方便预测框和图像的宽高进行相乘
        # -----------------------------------------------------------------#
        box_yx = box_xy[..., ::-1]  # 将 x, y 顺序交换为 y, x
        box_hw = box_wh[..., ::-1]  # 将 w, h 顺序交换为 h, w
        # input_shape = np.array(input_shape)  # 输入图像的尺度，通常是模型的输入尺寸
        image_shape = np.array(image_shape)  # 原始图像的尺度

        # 计算框的左上角和右下角坐标
        box_mins = box_yx - (box_hw / 2.)
        box_maxes = box_yx + (box_hw / 2.)

        # 将框的坐标从中心点和宽高形式转换为 [y_min, x_min, y_max, x_max] 形式
        boxes = np.concatenate([box_mins[..., 0:1], box_mins[..., 1:2],
                                box_maxes[..., 0:1], box_maxes[..., 1:2]], axis=-1)

        # -----------------------------------------------------------------#
        #   将框的坐标从输入图像尺度调整到原始图像尺度
        #   因为此前归一化是除以input_shape, scale = image.shape/input.shape
        # -----------------------------------------------------------------#
        boxes *= np.concatenate([image_shape, image_shape], axis=-1)

        #  boxes的格式为: [y_min, x_min, y_max, x_max]
        return boxes

    def forward(self, roi_cls_locs, roi_scores, rois, image_shape, input_shape, nms_iou=0.3, confidence=0.5):
        '''

        :param roi_cls_locs: 回归参数，用于调整候选框的坐标
        :param roi_scores: 候选框的类别置信度
        :param rois: 候选框的坐标
        :param image_shape: 原始图像的尺寸（如(1080, 1920)）
        :param input_shape: 模型输入图像的尺寸（如(640, 640)）
        :param nms_iou: 非极大抑制（NMS）的 IoU 阈值，默认为 0.3
        :param confidence: 置信度阈值，默认为 0.5
        :return: (y_min, x_min, y_max, x_max, confidence, class_id)
        '''
        results = []
        bs = len(roi_cls_locs)
        # --------------------------------#
        #   batch_size, num_rois, 4
        # --------------------------------#
        rois = rois.view((bs, -1, 4))
        # ----------------------------------------------------------------------------------------------------------------#
        #   对每一张图片进行处理，由于在predict.py的时候，我们只输入一张图片，所以for i in range(len(mbox_loc))只进行一次
        # ----------------------------------------------------------------------------------------------------------------#
        for i in range(bs):
            # ----------------------------------------------------------#
            #   对回归参数进行reshape
            # ----------------------------------------------------------#
            roi_cls_loc = roi_cls_locs[i] * self.std
            # ----------------------------------------------------------#
            #   第一维度是建议框的数量，第二维度是每个种类
            #   第三维度是对应种类的调整参数
            # ----------------------------------------------------------#
            roi_cls_loc = roi_cls_loc.view([-1, self.num_classes, 4])

            # -------------------------------------------------------------#
            #   利用classifier网络的预测结果对建议框进行调整获得预测框
            #   num_rois, 4 -> num_rois, 1, 4 -> num_rois, num_classes, 4
            # -------------------------------------------------------------#
            roi = rois[i].view((-1, 1, 4)).expand_as(roi_cls_loc)
            cls_bbox = loc2bbox(roi.contiguous().view((-1, 4)), roi_cls_loc.contiguous().view((-1, 4)))
            cls_bbox = cls_bbox.view([-1, (self.num_classes), 4])
            # -------------------------------------------------------------#
            #   对预测框进行归一化，调整到0-1之间
            #   在图像处理和计算机视觉中，图像的形状通常表示为 (height, width)
            # -------------------------------------------------------------#
            cls_bbox[..., [0, 2]] = (cls_bbox[..., [0, 2]]) / input_shape[1]
            cls_bbox[..., [1, 3]] = (cls_bbox[..., [1, 3]]) / input_shape[0]

            roi_score = roi_scores[i]
            prob = F.softmax(roi_score, dim=-1)

            results.append([])
            for c in range(1, self.num_classes):
                # --------------------------------#
                #   取出属于该类的所有框的置信度
                #   判断是否大于门限
                # --------------------------------#
                c_confs = prob[:, c]
                c_confs_m = c_confs > confidence

                if len(c_confs[c_confs_m]) > 0:
                    # -----------------------------------------#
                    #   取出得分高于confidence的框
                    # -----------------------------------------#
                    boxes_to_process = cls_bbox[c_confs_m, c]
                    confs_to_process = c_confs[c_confs_m]

                    keep = nms(
                        boxes_to_process,
                        confs_to_process,
                        nms_iou
                    )
                    # -----------------------------------------#
                    #   取出在非极大抑制中效果较好的内容
                    # -----------------------------------------#
                    good_boxes = boxes_to_process[keep]
                    confs = confs_to_process[keep][:, None]
                    labels = (c - 1) * torch.ones((len(keep), 1)).cuda() if confs.is_cuda else (c - 1) * torch.ones((len(keep), 1))
                    # -----------------------------------------#
                    #   将label、置信度、框的位置进行堆叠。
                    # -----------------------------------------#
                    c_pred = torch.cat((good_boxes, confs, labels), dim=1).cpu().numpy()
                    # 添加进result里
                    results[-1].extend(c_pred)

            if len(results[-1]) > 0:
                results[-1] = np.array(results[-1])
                # box_xy: 预测框的中心坐标；box_wh: 预测框的宽高
                box_xy, box_wh = (results[-1][:, 0:2] + results[-1][:, 2:4])/2, results[-1][:, 2:4] - results[-1][:, 0:2]
                results[-1][:, :4] = self.frcnn_correct_boxes(box_xy, box_wh, input_shape, image_shape)

        # 返回每张图像的检测结果，格式为(y_min, x_min, y_max, x_max, confidence, class_id)
        return results

