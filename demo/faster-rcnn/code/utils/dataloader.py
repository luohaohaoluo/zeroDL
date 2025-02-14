import cv2
import numpy as np
import torch

from torch.utils.data.dataset import Dataset
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageDraw, ImageFont
from utils.utilss import cvtColor, preprocess_input


class FRCNNDataset(Dataset):
    def __init__(self, annotation_lines, input_shape=[600, 600], train=True):
        self.annotation_lines = annotation_lines
        self.length = len(annotation_lines)
        self.input_shape = input_shape
        self.train = train

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        '''
        :param index: 索引
        :return: image, box:(左上点x坐标, 左上点y坐标, 右下点x坐标, 右下点y坐标), label
        '''

        index = index % self.length
        # ---------------------------------------------------#
        #   训练时进行数据的随机增强
        #   验证时不进行数据的随机增强
        # ---------------------------------------------------#
        image, y = self.get_random_data(self.annotation_lines[index], self.input_shape[0:2], random=self.train)
        image = np.transpose(preprocess_input(np.array(image, dtype=np.float32)), (2, 0, 1))
        box_data = np.zeros((len(y), 5))
        if len(y) > 0:
            box_data[:len(y)] = y

        box = box_data[:, :4]
        label = box_data[:, -1]
        return image, box, label

    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a

    def get_random_data(self, annotation_line, input_shape, jitter=.3, hue=.1, sat=0.7, val=0.4, random=True):
        line = annotation_line.split()
        # ------------------------------#
        #   读取图像并转换成RGB图像
        # ------------------------------#
        image = Image.open(line[0])
        image = cvtColor(image)
        # ------------------------------#
        #   获得图像的高宽与目标高宽
        # ------------------------------#
        iw, ih = image.size
        h, w = input_shape
        # ------------------------------#
        #   获得预测框
        # ------------------------------#
        box = np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]])

        if not random:

            # 保证缩放后的图片尺寸不会大于规定尺寸
            scale = min(w / iw, h / ih)
            nw = int(iw * scale)
            nh = int(ih * scale)
            # --------------------------------------------------------------------------#
            #   即便是reshape的size大于原size, dx，dy也不会是负值, 因为前面的min()确保了，
            #   取了最小的缩放因子, 顶天也就和要求的w或h一样，但是肯定不会超过
            #   dx, dy是为了让图片在灰色背景上保持居中
            # --------------------------------------------------------------------------#
            dx = (w - nw) // 2
            dy = (h - nh) // 2

            # ---------------------------------#
            #   将图像多余的部分加上灰条
            # ---------------------------------#
            image = image.resize((nw, nh), Image.BICUBIC)
            new_image = Image.new('RGB', (w, h), (128, 128, 128))
            new_image.paste(image, (dx, dy))
            image_data = np.array(new_image, np.float32)

            # ---------------------------------#
            #   对真实框进行调整
            # ---------------------------------#
            if len(box) > 0:
                np.random.shuffle(box)
                box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
                box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
                box[:, 0:2][box[:, 0:2] < 0] = 0
                box[:, 2][box[:, 2] > w] = w
                box[:, 3][box[:, 3] > h] = h
                box_w = box[:, 2] - box[:, 0]
                box_h = box[:, 3] - box[:, 1]
                box = box[np.logical_and(box_w > 1, box_h > 1)]  # discard invalid box

            return image_data, box

        # ------------------------------------------#
        #   对图像进行缩放并且进行长和宽的扭曲
        # ------------------------------------------#
        new_ar = iw / ih * self.rand(1 - jitter, 1 + jitter) / self.rand(1 - jitter, 1 + jitter)
        scale = self.rand(.25, 2)
        if new_ar < 1:
            nh = int(scale * h)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * w)
            nh = int(nw / new_ar)
        image = image.resize((nw, nh), Image.BICUBIC)

        # ------------------------------------------#
        #   将图像多余的部分加上灰条
        # ------------------------------------------#
        dx = int(self.rand(0, w - nw))
        dy = int(self.rand(0, h - nh))
        new_image = Image.new('RGB', (w, h), (128, 128, 128))
        new_image.paste(image, (dx, dy))
        image = new_image

        # ------------------------------------------#
        #   翻转图像
        # ------------------------------------------#
        flip = self.rand() < .5
        if flip: image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

        image_data = np.array(image, np.uint8)
        # ---------------------------------#
        #   对图像进行色域变换
        #   计算色域变换的参数
        # ---------------------------------#
        r = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
        # ---------------------------------#
        #   将图像转到HSV上
        # ---------------------------------#
        hue, sat, val = cv2.split(cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV))
        dtype = image_data.dtype
        # ---------------------------------#
        #   应用变换
        # ---------------------------------#
        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        image_data = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        image_data = cv2.cvtColor(image_data, cv2.COLOR_HSV2RGB)

        # ---------------------------------#
        #   对真实框进行调整
        # ---------------------------------#
        if len(box) > 0:
            np.random.shuffle(box)
            box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
            box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
            if flip: box[:, [0, 2]] = w - box[:, [2, 0]]
            box[:, 0:2][box[:, 0:2] < 0] = 0
            box[:, 2][box[:, 2] > w] = w
            box[:, 3][box[:, 3] > h] = h
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w > 1, box_h > 1)]

        return image_data, box


# DataLoader中collate_fn使用
def frcnn_dataset_collate(batch):
    images = []
    bboxes = []
    labels = []
    for img, box, label in batch:
        images.append(img)
        bboxes.append(box)
        labels.append(label)
    images = torch.from_numpy(np.array(images))
    return images, bboxes, labels


if __name__ == "__main__":
    train_annotation_path = '../2007_train.txt'
    with open(train_annotation_path, encoding='utf-8') as f:
        train_lines = f.readlines()

    label_map_path = "../dataset/voc_classes.txt"
    with open(label_map_path, encoding='utf-8') as f:
        label_lines = f.readlines()
    label_lines = [a.strip() for a in label_lines]

    input_shape = [600, 600]
    train_dataset = FRCNNDataset(train_lines, input_shape, train=True)
    # print(train_dataset)

    # test  --->  12, 19
    # train --->  26
    num = 26
    # print(train_dataset[num])
    print(train_dataset[num][1].shape)
    print(train_dataset[num][2].shape)


    img, boxes, real_label = train_dataset[num][0], train_dataset[num][1], train_dataset[num][2]
    # print(f"img.unique: {np.unique(img)}")

    # # 图像归一化处理
    im = Image.fromarray(img.dot(255).astype(np.uint8).transpose(1, 2, 0))
    draw = ImageDraw.Draw(im)

    # 获取图像大小
    image_width, image_height = im.size

    # 遍历所有的边界框和标签
    for boxe, label in zip(boxes, real_label):
        print(f"Boxe: {boxe}, Label: {label}")  # 输出每个边界框和标签

        # 检查 boxe 是否是合法的四元组
        if len(boxe) == 4:
            # 确保 boxe 不超出图像范围
            boxe = (max(0, min(boxe[0], image_width - 1)),
                    max(0, min(boxe[1], image_height - 1)),
                    max(0, min(boxe[2], image_width - 1)),
                    max(0, min(boxe[3], image_height - 1)))

            draw.rectangle(boxe, outline=(255, 0, 0), width=2)
            draw.rectangle((boxe[0], boxe[1] + 10, boxe[0] + 70, boxe[1]), outline=(255, 0, 0), fill=(255, 0, 0),
                           width=2)
            # 如果是这个str(label_lines[int(label)]), 就不正确了
            draw.text((boxe[0] + 2, boxe[1]), str(label_lines[int(float(label))]), fill=(255, 255, 255))
        else:
            print(f"无效的边界框: {boxe}")

    # 显示图像
    im.show()



