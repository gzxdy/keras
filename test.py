from keras.layers import *
from keras import Model
import config
import pickle
import numpy
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD, Adam, RMSprop
import model

model = model.lstm()
model.load_weights(config.model_path)

X_train,Y_train,X_test,Y_test = pickle.load(open("./data1","rb"))
result = model.predict(X_test)
#print(result)
Y_test = numpy.mat(Y_test)
result = numpy.argmax(result,axis=1)
#print(result)

Y_test_1 = numpy.argmax(numpy.array(Y_test),axis=1)
#print(Y_test)
count = 0
for i,j in zip(result,Y_test_1):
    if i == j:
        count+=1
print("acc: ",count/len(Y_test_1))

print(count,"---",len(Y_test_1))

model.compile(loss='categorical_crossentropy',  # 亦称作多类的对数损失，注意使用该目标函数时，需要将标签转化为形如(nb_samples, nb_classes)的二值序列
              optimizer=RMSprop(),  # optimizer是优化器(优化参数的算法),SGD(随机梯度下降)
              metrics=['accuracy'])  # metrics 性能评估函数类似与目标函数, 只不过该性能的评估结果并不会用于训练.
score = model.evaluate(X_test, Y_test, verbose=1)
print('loss:', score[0])
print('Test accuracy:', score[1])

