import numpy as np

from layers import *
from layer_utils import *


class VarLeNet(object):
    def __init__(self, input_dim=(1, 28, 28), filter_size=5, num_classes=10,
                 weight_scale=1e-3, reg=0.0,
                 dtype=np.float32):
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        # 第一层 conv - relu - 2x2 max pool
        # 零填充2 保证为32*32
        # 卷积核 6*5*5
        pad1 = 2
        num_filters1 = 6
        # 卷积后的输出大小
        H_conv1 = 1 + (input_dim[1] + 2 * pad1 - filter_size) / 1
        W_conv1 = 1 + (input_dim[2] + 2 * pad1 - filter_size) / 1
        # 池化后的输出大小
        H_pool1 = 1 + (H_conv1 - 2) / 2
        W_pool1 = 1 + (W_conv1 - 2) / 2

        # W1实现卷积层1
        self.params['W1'] = np.random.normal(0, weight_scale,
                                             size=(num_filters1, input_dim[0],
                                                   filter_size, filter_size))
        self.params['b1'] = np.zeros(num_filters1)

        # 第二层 conv - relu - 2x2 max pool
        # 卷积核 16*5*5
        # 接收池化1的输出
        pad2 = 0
        num_filters2 = 16
        # 卷积后的输出大小
        H_conv2 = 1 + (H_pool1 + 2 * pad2 - filter_size) / 1
        W_conv2 = 1 + (W_pool1 + 2 * pad2 - filter_size) / 1

        # 池化后的输出大小
        H_pool2 = 1 + (H_conv2 - 2) / 2
        W_pool2 = 1 + (W_conv2 - 2) / 2

        # W2实现卷积层2
        self.params['W2'] = np.random.normal(0, weight_scale,
                                             size=(num_filters2, num_filters1,
                                                   filter_size, filter_size))
        self.params['b2'] = np.zeros(num_filters2)

        # 第三层fc(16 * 5 * 5, 120) - relu
        # W3实现全连接
        self.params['W3'] = np.random.normal(0, weight_scale,
                                             size=(num_filters2 * int(H_pool2 * W_pool2), 120))
        self.params['b3'] = np.zeros(120)

        # 第四层fc(120，84) - relu
        # W4实现全连接
        self.params['W4'] = np.random.normal(0, weight_scale,
                                             size=(120, 84))
        self.params['b4'] = np.zeros(84)

        # 第五层fc(84,10)
        # W5实现评分函数
        self.params['W5'] = np.random.normal(0, weight_scale,
                                             size=(84, num_classes))
        self.params['b5'] = np.zeros(num_classes)

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    # 前向计算损失
    def loss(self, X, y=None):
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']
        W4, b4 = self.params['W4'], self.params['b4']
        W5, b5 = self.params['W5'], self.params['b5']
        scores = None
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
        # 前向传递

        # 第一层 conv - relu - 2x2 max pool
        conv_param1 = {'stride': 1, 'pad': 2}
        scores, cache1 = conv_relu_pool_forward(X, W1, b1, conv_param1, pool_param)
        # 第二层 conv - relu - 2x2 max pool
        conv_param2 = {'stride': 1, 'pad': 0}
        scores, cache2 = conv_relu_pool_forward(scores, W2, b2, conv_param2, pool_param)

        # 第三层fc(16 * 5 * 5, 120) - relu
        scores, cache3 = affine_relu_forward(scores, W3, b3)

        # 第四层fc(120，84) - relu
        scores, cache4 = affine_relu_forward(scores, W4, b4)

        # 第五层fc(84,10)
        scores, cache5 = affine_forward(scores, W5, b5)

        if y is None:
            return scores
        loss, grads = 0, {}
        loss, dx = softmax_loss(scores, y)
        # 正则化损失
        loss += 0.5 * self.reg * (np.sum(W1 ** 2) + np.sum(W2 ** 2) +
                                  np.sum(W3 ** 2) + np.sum(W4 ** 2) + np.sum(W5 ** 2))

        # 反向传播
        dx, dw5, db5 = affine_backward(dx, cache5)
        grads['W5'] = dw5 + self.reg * W5
        grads['b5'] = db5

        dx, dw4, db4 = affine_relu_backward(dx, cache4)
        grads['W4'] = dw4 + self.reg * W4
        grads['b4'] = db4

        dx, dw3, db3 = affine_relu_backward(dx, cache3)
        grads['W3'] = dw3 + self.reg * W3
        grads['b3'] = db3

        dx, dw2, db2 = conv_relu_pool_backward(dx, cache2)
        grads['W2'] = dw2 + self.reg * W2
        grads['b2'] = db2

        dx, dw1, db1 = conv_relu_pool_backward(dx, cache1)
        grads['W1'] = dw1 + self.reg * W1
        grads['b1'] = db1
        return loss, grads
