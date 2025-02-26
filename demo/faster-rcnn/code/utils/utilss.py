import random

import numpy as np
import torch
from PIL import Image

# ---------------------------------------------------------#
#   将图像转换成RGB图像，防止灰度图在预测时报错。
#   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
# ---------------------------------------------------------#
def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image
    else:
        image = image.convert('RGB')
        return image

# ---------------------------------------------------#
#   对输入图像进行resize
# ---------------------------------------------------#
def resize_image(image, size):
    w, h = size
    new_image = image.resize((w, h), Image.BICUBIC)
    return new_image

# ---------------------------------------------------#
#   获得类
# ---------------------------------------------------#
def get_classes(classes_path):
    with open(classes_path, encoding='utf-8') as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names, len(class_names)

# ---------------------------------------------------#
#   获得学习率
# ---------------------------------------------------#
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

# ---------------------------------------------------#
#   设置固定种子保证结果的可复现性
# ---------------------------------------------------#
def seed_everything(seed=11):
    random.seed(seed)                           # 设置 Python自带的random模块的随机种子
    np.random.seed(seed)                        # 设置NumPy库的随机种子
    torch.manual_seed(seed)                     # 设置 PyTorch 库的 CPU 随机种子
    torch.cuda.manual_seed(seed)                # 设置 PyTorch 库的单个 GPU 随机种子
    torch.cuda.manual_seed_all(seed)            # 设置 PyTorch 库的单个 GPU 随机种子
    torch.backends.cudnn.deterministic = True   # cuDNN 的确定性模式为开启状态
    torch.backends.cudnn.benchmark = False      # 关闭基准测试模式后，cuDNN 会使用固定的算法


# ---------------------------------------------------#
#   设置Dataloader的种子
# ---------------------------------------------------#
def worker_init_fn(worker_id, rank, seed):
    worker_seed = rank + seed
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def preprocess_input(image):
    image /= 255.0
    return image


def show_config(**kwargs):
    print('Configurations:')
    print('-' * 70)
    print('|%25s | %40s|' % ('keys', 'values'))
    print('-' * 70)
    for key, value in kwargs.items():
        print('|%25s | %40s|' % (str(key), str(value)))
    print('-' * 70)

def get_new_img_size(height, width, img_min_side=600):
    if width <= height:
        f = float(img_min_side) / width
        resized_height = int(f * height)
        resized_width = int(img_min_side)
    else:
        f = float(img_min_side) / height
        resized_width = int(f * width)
        resized_height = int(img_min_side)

    return resized_height, resized_width