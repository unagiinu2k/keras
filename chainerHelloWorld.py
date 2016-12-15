import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions

x_data = np.array([5], dtype=np.float32)
x = Variable(x_data)
z_data = np.array([5], dtype=np.float32)
z = Variable(z_data)
#y = x + 1
y = x + 2 * z
if False:
    x.data
    y.data


w = z ** 2
#w.backward()
w.backward(retain_grad = True)
x.grad
y.grad
z.grad



y.backward()
#Note that gradients are accumulated by the method rather than overwritten. So first you must clear gradients to renew the computation. It can be done by calling the cleargrads()

x.grad # yのxでの偏微分を返す
z.grad #yのzでの偏微分を返す













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


train_iter = iterators.SerialIterator(train, batch_size=100, shuffle=True)#training processのミニバッチサイズなどを与える
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


class Classifier(Chain):
    def __init__(self, predictor):
        super(Classifier, self).__init__(predictor=predictor)
    def __call__(self, x, t):
        y = self.predictor(x)
        loss = F.softmax_cross_entropy(y, t)
        accuracy = F.accuracy(y, t)
        report({'loss': loss, 'accuracy': accuracy}, self)
        return loss



model = Classifier(MLP())#model is a "function" which returns error between MLPmodel and the correct answer


optimizer = optimizers.SGD()#誤差拡散法を使う
optimizer.setup(model) #目的関数を与える
updater = training.StandardUpdater(train_iter , optimizer)
trainer = training.Trainer(updater, (20, 'epoch'), out='result')
if False:

    trainer.extend(extensions.Evaluator(test_iter, model))
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(['epoch', 'main/accuracy', 'validation/main/accuracy']))
    trainer.extend(extensions.ProgressBar())
trainer.run()