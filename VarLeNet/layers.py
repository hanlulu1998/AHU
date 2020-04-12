import numpy as np

'''
@Descripttion: 神经网络各个层
@version: 1.0
@Author: Han Lulu
@Date: 2020-04-11 10:07:17
@LastEditors: Han Lulu
@LastEditTime: 2020-04-11 10:34:07
'''


# 仿射层（全连接层）
def affine_forward(x, w, b):  # 前向
    out = None
    # f=w*x+b
    out = x.reshape(x.shape[0], -1).dot(w) + b
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):  # 反向
    x, w, b = cache
    dx, dw, db = None, None, None
    # 反向传播求梯度
    dx = dout.dot(w.T).reshape(x.shape)
    dw = x.reshape(x.shape[0], -1).T.dot(dout)
    db = np.sum(dout, axis=0)
    return dx, dw, db


# 激活函数
def relu_forward(x):  # 前向
    out = None
    # 使用简单的maxium即可
    out = np.maximum(0, x)
    cache = x
    return out, cache


def relu_backward(dout, cache):  # 反向
    dx, x = None, cache
    # 仅有大于0的有贡献，其他为0
    dx = (x > 0) * dout
    return dx


# 卷积层
def conv_forward(x, w, b, conv_param):  # 前向
    out = None
    stride = conv_param['stride']
    pad = conv_param['pad']
    # F是卷积核数目（即输出的深度），C是通道数（如RGB三通道）
    N, C, H, W = x.shape
    F, C, HH, WW = w.shape
    # 输出形状
    H_out = int(1 + (H + 2 * pad - HH) / stride)
    W_out = int(1 + (W + 2 * pad - WW) / stride)
    out = np.zeros((N, F, H_out, W_out))
    # 零填充
    x_pad = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)),
                   mode='constant',
                   constant_values=0)
    for j in range(H_out):
        for k in range(W_out):
            # 按照stride步长依次选择X_pad中卷积核大小的区域
            x_mask = x_pad[:, :, j * stride:j * stride + HH, k *
                                                             stride:k * stride + WW]
            for i in range(F):
                # 进行卷积操作
                out[:, i, j, k] = np.sum(x_mask * w[i], axis=(1, 2, 3))
    # 扩展维度，使b变成4维数据和out对应
    out += b[None, :, None, None]
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward(dout, cache):  # 反向
    dx, dw, db = None, None, None
    x, w, b, conv_param = cache
    stride = conv_param['stride']
    pad = conv_param['pad']
    _, _, HH, WW = w.shape
    N, F, H_out, W_out = dout.shape
    # 零填充
    x_pad = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)),
                   mode='constant',
                   constant_values=0)
    dx = np.zeros_like(x)
    dx_pad = np.zeros_like(x_pad)
    dw = np.zeros_like(w)
    # db保证和形状一致
    db = np.sum(dout, axis=(0, 2, 3))

    for i in range(H_out):
        for j in range(W_out):
            x_mask = x_pad[:, :, i * stride:i * stride + HH, j *
                                                             stride:j * stride + WW]
            # dw为dout和x_mask的卷积
            for k in range(F):
                # 固定i,j,k后dout输出的是N数目的一维向量，需要拓展维度使其和x_mask对应
                # (N,C,H_core,W_core)*(N,1,1,1)=(N,C,H_core,W_core)
                dw[k, :, :, :] += np.sum(
                    (dout[:, k, i, j])[:, None, None, None] * x_mask, axis=0)
            # dx_pad为dout和w的卷积
            for n in range(N):
                # 固定i,j,n后dout输出的是F数目的一维向量，需要拓展维度使其和w对应
                dx_pad[n, :, i * stride:i * stride + HH, j * stride:j * stride + WW] = \
                    np.sum((dout[n, :, i, j])[:, None, None, None] * w, axis=0) + \
                    dx_pad[n, :,
                    i * stride:i * stride + HH,
                    j * stride:j * stride + WW]
    # 需要舍弃填充的部分
    dx = dx_pad[:, :, pad:-pad, pad:-pad]
    return dx, dw, db


# 最大池化层
def max_pool_forward(x, pool_param):  # 前向
    out = None
    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_width']
    stride = pool_param['stride']
    N, C, H, W = x.shape
    H_out = int(1 + (H - pool_height) / stride)
    W_out = int(1 + (W - pool_width) / stride)
    out = np.zeros((N, C, H_out, W_out))

    for i in range(H_out):
        for j in range(W_out):
            x_mask = x[:, :, i * stride:i * stride + pool_height, j *
                                                                  stride:j * stride + pool_width]
            # 找到（:,:,H,W）也就是axis=(2, 3)最大值
            out[:, :, i, j] = np.max(x_mask, axis=(2, 3))
    cache = (x, pool_param)
    return out, cache


def max_pool_backward(dout, cache):  # 反向
    dx = None
    x, pool_param = cache
    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_width']
    stride = pool_param['stride']
    N, C, H_out, W_out = dout.shape
    dx = np.zeros_like(x)

    for i in range(H_out):
        for j in range(W_out):
            x_mask = x[:, :, i * stride:i * stride + pool_height, j *
                                                                  stride:j * stride + pool_width]
            # 这里找到最大值的位置，最大值矩阵是二维需要拉伸到四维和x_mask保持一致
            mask = (x_mask == np.max(x_mask, axis=(2, 3))[:, :, None, None])
            # 反向只有mask中最大位置有贡献，其余为0
            dx[:, :, i * stride:i * stride + pool_height, j * stride:j * stride + pool_width] = \
                dout[:, :, i, j][:, :, None, None] * mask
    return dx


# 交叉熵损失
def softmax_loss(x, y):
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    N = x.shape[0]
    loss = -np.sum(log_probs[np.arange(N), y]) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx


# 拉伸展开层
def flatten(x):
    N, C, H, W = x.shape
    cache = x
    return x.reshape(N, -1), cache


# 拉伸展开层反向
def unflatten(dout, cache):
    N, C, H, W = cache.shape
    dx = dout.reshape(N, C, H, W)
    return dx
