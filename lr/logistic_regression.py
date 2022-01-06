import argparse
import os
import random
import time
from datetime import datetime
from pathlib import Path
import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


class LogisticRegression(object, opt=argparse.ArgumentParser()):
    """
    逻辑回归算法，opt传入默认参数
    """
    def __init__(self):
        self.alpha = opt.alpha
        self.penalty = opt.penalty
        self.tol = opt.tol
        self.C = opt.C
        self.max_iter = opt.max_iter

    def load_data(self):
        return

    def grad_descent(self, ):
        return

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
