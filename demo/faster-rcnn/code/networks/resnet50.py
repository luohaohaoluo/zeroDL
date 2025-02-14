import math

import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url

# 残差块
class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        """
        :param inplanes: 输入通道
        :param planes:   输出通道
        :param stride:
        :param downsample:
        """
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        """
        :param block: 残差块
        :param layers: 一个列表，对于每一层，每个元素代表残差块循环多少次
        :param num_classes: 预测的类别
        """

        # -----------------------------------#
        #   假设输入进来的图片是224,224,3
        # -----------------------------------#
        self.inplanes = 64
        super(ResNet, self).__init__()

        # 3,224,224 -> 64,112,112
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # 64,112,112 -> 64,56,56
        # ceil_mode=True 将保存不足为kernel_size大小的数据保留，并参与计算
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)

        # 64,56,56 -> 256,56,56
        self.layer1 = self._make_layer(block, 64, layers[0])
        # 256,56,56 -> 512,28,28
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        # 512,28,28 -> 1024,14,14 到这里可以获得一个14,14,1024的共享特征层
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        # self.layer4被用在classifier模型中
        # 1024,14,14 -> 2048,7,7
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # 2048,7,7 -> 2048,1,1
        self.avgpool = nn.AvgPool2d(7)
        # 2048,1,1 -> num_classes,1,1
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        """

        :param block: 残差块
        :param planes: 这里其实是每次输入的通道数
        :param blocks: 这里更应该是循环 (blocks-1)次的不需要downsample的残差块
        :param stride: 步长
        :return: 返回的是每一层网络，里面包含了blocks次数的残差块
        """
        downsample = None
        # -------------------------------------------------------------------#
        #   当模型需要进行高和宽的压缩的时候，就需要用到残差边的downsample
        # -------------------------------------------------------------------#
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # print(f"conv1: {x.shape}")

        x = self.maxpool(x)
        # print(f"maxpool: {x.shape}")

        x = self.layer1(x)
        # print(f"conv2_x: {x.shape}")
        x = self.layer2(x)
        # print(f"conv3_x: {x.shape}")
        x = self.layer3(x)
        # print(f"conv4_x: {x.shape}")
        x = self.layer4(x)
        # print(f"conv5_x: {x.shape}")

        x = self.avgpool(x)
        # print(f"avgpool: {x.shape}")
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        # print(f"fc: {x.shape}")
        return x


def resnet50(pretrained=False):
    model = ResNet(Bottleneck, [3, 4, 6, 3])
    if pretrained:
        state_dict = load_state_dict_from_url("https://download.pytorch.org/models/resnet50-19c8e357.pth",
                                              model_dir="./model_data")
        model.load_state_dict(state_dict)
    # ----------------------------------------------------------------------------#
    #   获取特征提取部分，从conv1到model.layer3，最终获得一个14,14,1024的特征层
    # ----------------------------------------------------------------------------#
    features = list([model.conv1, model.bn1, model.relu, model.maxpool, model.layer1, model.layer2, model.layer3])
    # ----------------------------------------------------------------------------#
    #   获取分类部分，从model.layer4到model.avgpool
    # ----------------------------------------------------------------------------#
    classifier = list([model.layer4, model.avgpool])

    features = nn.Sequential(*features)
    classifier = nn.Sequential(*classifier)
    return features, classifier


if __name__ == "__main__":
    features, classifier = resnet50(pretrained=True)
    x = torch.rand((1, 3, 224, 224))
    y1 = features(x)
    y = classifier(y1)
    # y = model(x)
    print(y1.shape)
    print(y.shape)
    # print(model)


