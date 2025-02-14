# 目标检测
利用VOC2007数据集，参考Bubbliiiing大佬的[faster-rcnn](https://github.com/bubbliiiing/faster-rcnn-pytorch)训练并且学习整个流程，但是在大佬源代码中我发现有一处错误，就是在预测的时候，还原锚框尺寸比例的代码出了错误。

在code文件夹下是整个工程代码。其中，dataset存放VOC2007数据集和voc的类别文件；model_data存放预训练权重，包括resent50以及训练好的模型权重；networks存放faster-rcnn网络的模型；utils存放一些需要的工具包

对整个工程的讲解可以查看**fasterrcnn.md**文件，并支持pdf版本下载，根据代码讲解更好地去理解双端目标检测网络的运行。

这是我运行时的参数配置，主要就是epoch=100，pretrained=False，Freeze_Train=False
<p align = "center">  
<img src=./picture/training.png  width="900"/>
</p>

最后的map为

运行predicy.py, 选择模式1单张预测，可得到以下结果
<p align = "center">  
<img src=./picture/test.png  width="700"/>
</p>

感兴趣的朋友可以试试pretrained=True的冻结训练情况，应该是要比现在好得多，毕竟backbone见多识广是有好处的
