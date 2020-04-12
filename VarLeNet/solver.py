import pickle as pickle

import numpy as np

import optim


# 求解器
class Solver(object):
    def __init__(self, model, data, **kwargs):
        self.model = model  # 定义的网络模型
        # data类型
        '''
        data = {
        'X_train': # 训练集
        'y_train': # 训练标签
        'X_val': # 验证集
        'y_val': # 验证标签
        }
        '''
        self.X_train = data['X_train']
        self.y_train = data['y_train']
        self.X_val = data['X_val']
        self.y_val = data['y_val']

        self.update_rule = kwargs.pop('update_rule', 'sgd')  # 优化器，默认SGD
        self.optim_config = kwargs.pop('optim_config', {})  # 优化器参数
        self.lr_decay = kwargs.pop('lr_decay', 1.0)  # 学习衰减率 默认1
        self.batch_size = kwargs.pop('batch_size', 100)  # batch_size 默认100
        self.num_epochs = kwargs.pop('num_epochs', 10)  # epochs 默认10
        self.num_train_samples = kwargs.pop('num_train_samples', 100)
        self.num_val_samples = kwargs.pop('num_val_samples', None)
        self.print_every = kwargs.pop('print_every', 10)  # 每迭代多少次打印 默认10
        self.verbose = kwargs.pop('verbose', True)  # 是否打印显示 默认显示

        # 如果有额外的关键字参数，则产生错误
        if len(kwargs) > 0:
            extra = ', '.join('"%s"' % k for k in list(kwargs.keys()))
            raise ValueError('Unrecognized arguments %s' % extra)

        # 确保更新规则存在，然后用实际的函数替换字符串名称
        if not hasattr(optim, self.update_rule):
            raise ValueError('Invalid update_rule "%s"' % self.update_rule)
        self.update_rule = getattr(optim, self.update_rule)
        self._reset()

    # 初始化重置
    def _reset(self):
        self.epoch = 0
        self.best_val_acc = 0
        self.best_params = {}
        self.loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []
        self.optim_configs = {}
        for p in self.model.params:
            d = {k: v for k, v in self.optim_config.items()}
            self.optim_configs[p] = d

    # 单步梯度更新，由train()调用
    def _step(self):
        # 制作一个小批量的训练数据
        num_train = self.X_train.shape[0]
        batch_mask = np.random.choice(num_train, self.batch_size)
        X_batch = self.X_train[batch_mask]
        y_batch = self.y_train[batch_mask]

        # 计算梯度和损失
        loss, grads = self.model.loss(X_batch, y_batch)
        self.loss_history.append(loss)

        # 执行参数更新
        for p, w in self.model.params.items():
            dw = grads[p]
            config = self.optim_configs[p]
            next_w, next_config = self.update_rule(w, dw, config)
            self.model.params[p] = next_w
            self.optim_configs[p] = next_config

    # 检查精度
    def check_accuracy(self, X, y, num_samples=None, batch_size=100):
        N = X.shape[0]
        # 可能是样本的子数据
        if num_samples is not None and N > num_samples:
            mask = np.random.choice(N, num_samples)
            N = num_samples
            X = X[mask]
            y = y[mask]
        # 分批计算预测
        num_batches = N // batch_size
        if N % batch_size != 0:
            num_batches += 1
        y_pred = []
        for i in range(num_batches):
            start = i * batch_size
            end = (i + 1) * batch_size
            scores = self.model.loss(X[start:end])
            y_pred.append(np.argmax(scores, axis=1))
        y_pred = np.hstack(y_pred)
        acc = np.mean(y_pred == y)
        return acc

    # 训练函数
    def train(self):
        num_train = self.X_train.shape[0]
        iterations_per_epoch = max(num_train // self.batch_size, 1)
        num_iterations = self.num_epochs * iterations_per_epoch

        # 多次迭代执行单步更新
        for t in range(num_iterations):
            self._step()
            # 打印训练损失
            if self.verbose and t % self.print_every == 0:
                print('(Iteration %d / %d) loss: %f' %
                      (t + 1, num_iterations, self.loss_history[-1]))

            # 在每个epoch结束时，增加epoch计数器和衰变学习率
            epoch_end = (t + 1) % iterations_per_epoch == 0
            if epoch_end:
                self.epoch += 1
                for k in self.optim_configs:
                    self.optim_configs[k]['learning_rate'] *= self.lr_decay
            # 每个epoch结束时检查训练集和验证集的精确度。
            first_it = (t == 0)
            last_it = (t == num_iterations - 1)
            if first_it or last_it or epoch_end:
                train_acc = self.check_accuracy(
                    self.X_train,
                    self.y_train,
                    num_samples=self.num_train_samples)
                val_acc = self.check_accuracy(self.X_val,
                                              self.y_val,
                                              num_samples=self.num_val_samples)
                self.train_acc_history.append(train_acc)
                self.val_acc_history.append(val_acc)

                if self.verbose:
                    print('(Epoch %d / %d) train acc: %f; val_acc: %f' %
                          (self.epoch, self.num_epochs, train_acc, val_acc))

                # 迭代更新模型
                if val_acc > self.best_val_acc:
                    self.best_val_acc = val_acc
                    self.best_params = {}
                    for k, v in self.model.params.items():
                        self.best_params[k] = v.copy()
        self.model.params = self.best_params
