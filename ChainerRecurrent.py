import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions

l = L.LSTM(100 , 50)
l.reset_state()

x1 = Variable(np.random.randn(10, 100).astype(np.float32))
x1.data.shape # 10x100行列。１００次元ベクトル１０サンプルのイメージ？
x2 = Variable(np.random.randn(10, 100).astype(np.float32))

l.reset_state()
tmp = l(x1)
tmp2 = l(x2)
l.reset_state()

tmp2b = l(x2)　#tmp2とは異なる！
tmp2b.data.shape　#10x50 行列。５０次元ベクトル１０サンプル


class RNN(Chain):
    def __init__(self):
        super(RNN, self).__init__(
            embed=L.EmbedID(1000, 100),  # word embedding
            mid=L.LSTM(100, 50),  # the first LSTM layer
            out=L.Linear(50, 1000),  # the feed-forward output layer
        )

    def reset_state(self):
        self.mid.reset_state()

    def __call__(self, cur_word):
        # Given the current word ID, predict the next word.
        x = self.embed(cur_word)
        h = self.mid(x)
        y = self.out(h)
        return y

rnn = RNN()
model = L.Classifier(rnn)
optimizer = optimizers.SGD()
optimizer.setup(model)

def compute_loss(x_list):
    loss = 0
    for cur_word, next_word in zip(x_list, x_list[1:]):
        loss += model(cur_word, next_word)
    return loss