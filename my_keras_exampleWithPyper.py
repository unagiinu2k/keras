import pyper
import numpy
import pandas
dataSwitch = "random"
Rectify = "positive"

if dataSwitch == "random":
    from numpy.random import *

    X = randn(20000,5)
    Coefs = numpy.array([1,2,3,4,5])
    type(Coefs)
    y = X.dot(Coefs)
    y.shape
    N = 10000

    if Rectify == "positive":
        y = numpy.maximum(0 , y)
    elif Rectify == "nonlinear2":
        #z = y.copy()
        y[X[:,1]>0] = 0
        pandas.DataFrame(z).describe()
        pandas.DataFrame(y).describe()
#         pandas.DataFrame(numpy.where(X[: , 1] > 0 , y , 0)).describe()

    noise_sd = 0.1
    y = y + randn(y.shape[0]) * noise_sd


    if False:
        X = pandas.DataFrame(X)
        y = pandas.DataFrame(y)
        X.columns = ["a" , "b" , "c" , "d" , "e"]
        X.describe()
        y.columns = ["y"]

elif dataSwitch == "iris":

    r = pyper.R(use_pandas=True)
    iris = r.get("iris")
    iris.shape
    iris.columns = ["Sepal.Length" , "Sepal.Width" , "Petal.Length" , "Petal.Width" , "Species"]
    iris.loc[:  , ["Species"]]
    iris[["Sepal.Length"]]
    X = iris[["Sepal.Length" , "Sepal.Width" , "Petal.Length" ]]
    Y = iris[["Petal.Width"]]
    #Y = iris[["Species"]]
    X_train = X[0:120]
    X_test = X[121:]
    y_train = Y[0:120]
    y_test = Y[121:]
elif dataSwitch == "worldmap":
    r.run("library(ggplot2)")
    r.run("library(dplyr)")
    r.run("tmp = map_data('world')")
    worldmap = r.get("worldmap")#%>% select(-subregion)")
    worldmap.shape
    worldmap.columns = ["long" , "lat" , "group" , "order" , "region" , "subregion"]
    X = worldmap[["long" , "lat"]]
    y = worldmap[["region"]]
    N = 100000


X_train = X[0:N]
y_train = y[0:N]
X_test = X[(N+1):]
y_test = y[(N+1):]
from keras.models import Sequential


#from scipy.spatial.kdtree import innernode


from keras.layers.core import Dense, Activation

#https://github.com/amacbee/jupyter-notebooks/blob/master/introduction-keras/IMDB%20-%20Keras.ipynb


from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
#from keras.layers.recurrent import LSTM


#https://github.com/fchollet/keras/issues/108
model = Sequential()

#model.add(Dropout(0.5))#中間素子の５０％をランダムに無効化　http://olanleed.hatenablog.com/entry/2013/12/03/010945
#model.add(Dense(1 , input_dim=3 , init="uniform" , activation="linear"))#通常のfully connected NN layer

#model.add(Activation("softmax"))
if Rectify == "positive":
    model.add(Dense(output_dim = 1 , input_dim=X.shape[1]))#通常のfully connected NN layer
    model.add(Activation("relu"))
elif Rectify == "nonlinear2":
    #
    #https://github.com/fchollet/keras/issues/96
    model.add(Dropout(0.5 , input_shape= (X.shape[1] , )))
    model.add(Dense(output_dim = 4 , input_dim=X.shape[1]))#通常のfully connected NN layer
    #model.add(Activation("tanh"))
    model.add(Activation("relu"))
    model.add(Dense(output_dim = 4))
    model.add(Activation("tanh"))
    model.add(Dropout(0.5))
    model.add(Dense(output_dim = 1))
    model.add(Activation("linear"))


else:
    model.add(Dense(output_dim = 1 , input_dim=X.shape[1]))#通常のfully connected NN layer
    model.add(Activation("linear"))

#model.compile(loss = "categorical_crossentropy" , optimizer="rmsprop" , metrics  = ['accuracy'])
#model.compile(loss = "mean_squared_error" , optimizer="SGD" , metrics  = ['accuracy'])
model.compile(loss = "mean_squared_error" , optimizer="Adam" , metrics  = ['accuracy'])
#Adam以外はうまくいかない・・
#学習測度の階層ごとの決め方などの流儀の模様（？）

batch_size = 32
model.fit(X_train, y_train , batch_size = batch_size , nb_epoch = 40)# , batch_size = 10) # , batch_size = 50 , nb_epoch=3)

result = model.evaluate(X_test , y_test)# , batch_size=10)#http://qiita.com/nzw0301/items/d515ebf8e4b75f35bbcb
print(result)#最初の数値がcompile時のlossファンクションの数値。二番目の数値が精度。
Y2 = model.predict(X_test)
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(Y2 , y_test)

#import pandas
#pandas.DataFrame({'predicted' : Y2 , 'original' : y_test})
