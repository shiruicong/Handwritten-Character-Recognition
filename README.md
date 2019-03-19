# Handwritten-Character-Recognition
Handwritten Character Recognition Using LeNet Network

使用LeNet网络进行手写字体识别

使用的是公开数据集minist，28*28的图片，但是数据初始是1*784的结构，因此在进入网络之前需要先reshape一下，得到（28*28）的图像

网络结构大致为：卷积-池化-卷积-池化-全连接-全连接-softmax层，具体卷积核的大小和全连接的单元个数参看Lenet.py。

经测试，正确率达到96%

demo.py:输入一张28*28的灰度图像，可以识别出手写数字
