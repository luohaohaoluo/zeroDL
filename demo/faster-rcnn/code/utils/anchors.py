import numpy as np


'''
base_size    ---->  用于定义基础先验框的大小
feat_stride  ---->  用于确定这些先验框在特征图上的分布


'''

# --------------------------------------------#
#   1. 生成基础的先验框
# --------------------------------------------#
def generate_anchor_base(base_size=16, ratios=[0.5, 1, 2], anchor_scales=[8, 16, 32]):
    anchor_base = np.zeros((len(ratios) * len(anchor_scales), 4), dtype=np.float32)
    # --------------------------------------------#
    #   一共生成9个基础的先验框
    # --------------------------------------------#
    for i in range(len(ratios)):
        for j in range(len(anchor_scales)):
            h = base_size * anchor_scales[j] * np.sqrt(ratios[i])
            w = base_size * anchor_scales[j] * np.sqrt(1. / ratios[i])

            index = i * len(anchor_scales) + j

            # --------------------------------------------#
            #  每个基础先验框的格式：取宽和高的中心点展开
            # --------------------------------------------#

            anchor_base[index, 0] = - h / 2.
            anchor_base[index, 1] = - w / 2.
            anchor_base[index, 2] = h / 2.
            anchor_base[index, 3] = w / 2.
    return anchor_base


# --------------------------------------------#
#   2. 对基础先验框进行拓展对应到所有特征点上
# --------------------------------------------#
def _enumerate_shifted_anchor(anchor_base, feat_stride, height, width):
    # -------------------------------------------#
    #   计算网格中心点，其实是也是设置先验框的平移
    # -------------------------------------------#
    shift_x = np.arange(0, width * feat_stride, feat_stride)
    shift_y = np.arange(0, height * feat_stride, feat_stride)

    # ---------------------------------------------------------------#
    #   meshgrid():成一个网格, x沿着y方向扩展, y沿着x方向扩展
    #   ravel(): 展平成一维
    #   stack(): 按某一维度堆起来
    # ---------------------------------------------------------------#
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    # shift_x: (38, 38), shift_y: (38, 38)
    # print("shift_x:", shift_x.shape)
    # print("shift_y:", shift_y.shape)

    # -------------------------------------------------------#
    #   这里其实是通过平移的方式，改变先验框的左上点和右下点坐标
    # -------------------------------------------------------#
    shift = np.stack((shift_x.ravel(), shift_y.ravel(), shift_x.ravel(), shift_y.ravel(),), axis=1)
    # shift = np.stack((shift_x.ravel() - feat_stride / 2, shift_y.ravel() - feat_stride / 2,
    #                   shift_x.ravel() + feat_stride / 2, shift_y.ravel() + feat_stride / 2), axis=1)

    # print("shift:", shift.shape) # (1444, 4)

    # ---------------------------------#
    #   每个网格点上的9个先验框
    #   A:9 K: 1444(38 * 38)
    # ---------------------------------#
    A = anchor_base.shape[0]
    K = shift.shape[0]
    anchor = anchor_base.reshape((1, A, 4)) + shift.reshape((K, 1, 4))

    # ---------------------------------#
    #   所有的先验框
    # ---------------------------------#
    anchor = anchor.reshape((K * A, 4)).astype(np.float32)
    return anchor


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    nine_anchors = generate_anchor_base()
    print(nine_anchors)

    height, width, feat_stride = 38, 38, 16
    anchors_all = _enumerate_shifted_anchor(nine_anchors, feat_stride, height, width)
    print(np.shape(anchors_all))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.ylim(-300, 900)
    plt.xlim(-300, 900)
    shift_x = np.arange(0, width * feat_stride, feat_stride)
    shift_y = np.arange(0, height * feat_stride, feat_stride)
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    plt.scatter(shift_x, shift_y)
    box_widths = anchors_all[:, 2] - anchors_all[:, 0]
    box_heights = anchors_all[:, 3] - anchors_all[:, 1]

    for i in [108, 109, 110, 111, 112, 113, 114, 115, 116]:
        rect = plt.Rectangle([anchors_all[i, 0], anchors_all[i, 1]], box_widths[i], box_heights[i], color="r",
                             fill=False)
        ax.add_patch(rect)
    plt.show()
