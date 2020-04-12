# 毕设项目

## 题目：基于现代可重构计算平台的ANN架构

## VarLeNet

基于`LeNet5`的简化版CNN，就是去除了`LeNet5`中的局部对应，用于`MINST`数据集的识别，没有细调，使用`Pytorch`搭建的`VarLeNet`准确率在**99%**左右。

## 计算语言描述

参看`VarLeNet`文件夹里的源代码，提供Naive版和Pytorch版，即纯手写和使用`Pytorch`架构两种方式，具体实现见`VarLeNet_Naive.ipynb`和`VarLeNet_Pytorch.ipynb`。Naive版供浏览基本原理，代码实现没有考虑资源及优化问题，实现大规模训练很耗资源。Pytorch版提供`GPU/CPU`两种使用方式，可以进行大规模训练。

## FPGA构架

待更新...