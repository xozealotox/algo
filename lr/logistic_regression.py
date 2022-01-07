import argparse
import os
import random
import time
from datetime import datetime
from pathlib import Path
import numpy as np
from utils.general import LOGGER


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


class LogisticRegression(object):
    """
    逻辑回归算法，opt传入默认参数
    """

    def __init__(self, opt: argparse.ArgumentParser()):
        self.alpha = opt.alpha
        self.penalty = opt.penalty
        self.tol = opt.tol
        self.C = opt.C
        self.max_iter = opt.max_iter

    def load_data(self):
        return

    def grad_descent(self, data_mat_in, labels):
        """
        通过梯度下降找到局部最优的权重向量
        :param data_mat_in: 输入m*n矩阵 x
        :param labels: 结果m*1向量y
        :return: weights: n*1向量
        """
        data_mat = np.mat(data_mat_in)  # (m,n)矩阵
        label_mat = np.mat(labels).transpose()
        m, n = data_mat.shape()
        weights = np.ones((n, 1), dtype=np.float32)  # (n, 1)权重

        alpha = self.alpha  # 步长
        max_iter = self.max_iter
        loss = 1
        current_iter = 0

        while loss > self.tol and current_iter < max_iter:
            current_iter += 1
            h = sigmoid(data_mat * weights)  # 分类预测
            weights = weights + alpha * data_mat.transpose() * (label_mat - h)
            loss = np.sum(np.square(h - label_mat))
            LOGGER.info(f"Grident Decendent Step {current_iter}. Current Loss is:{loss} ")

        if current_iter < max_iter:
            LOGGER.info(f"Early Stop at {current_iter} iteration. Final Loss is {loss}")
        else:
            LOGGER.info(f"Iteration finished. Finial loss is {loss}")

        return weights

    def fit(self):
        return

    def predict(self):
        return

    def predict_proba(self):
        return


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha', type=float, default=0.01, help='learning rate of the algorithm')
    parser.add_argument('--penalty', type=str, default='l2', help='norm used in the penalization. l2 and l1')
    parser.add_argument('--tol', type=float, default=1e-4, help='tolerance for stopping criteria')
    parser.add_argument('--C', type=float, default=1.0,
                        help='inverse fo regularization strength. smaller means stronger regularization. must be a '
                             'positive float.')
    parser.add_argument('--max_iter', type=int, default=100, help='max iteration round')
    return parser.parse_known_args()[0] if known else parser.parse_args()


def main(opt):
    opt = parse_opt(True)
    lr = LogisticRegression(opt)


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
