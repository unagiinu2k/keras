import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions

train, test = datasets.get_mnist()

type(test)
type(test[0])
len(test[0])
type(test[0][0])
test[0][0].shape
test[0][1]


train_iter = iterators.SerialIterator(train, batch_size=100, shuffle=True)
type(train_iter)
if False:
    train_iter.next()

test_iter = iterators.SerialIterator(test, batch_size=100, repeat=False, shuffle=False)

class MLP(Chain):
    def __init__(self):
        super(MLP, self).__init__(
            l1=L.Linear(784, 100),
            l2=L.Linear(100, 100),
            l3=L.Linear(100, 10),
        )
    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        y = self.l3(h2)
        return y