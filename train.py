from keras.layers import *
from keras import Model
import config
import pickle
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD, Adam, RMSprop
import model
#from keras.losses import CategoricalCrossentropy

X_train,Y_train,X_test,Y_test = pickle.load(open("./data1","rb"))
model = model.lstm()

model.summary()

# 训练和评估
model.compile(loss='categorical_crossentropy',  # 亦称作多类的对数损失，注意使用该目标函数时，需要将标签转化为形如(nb_samples, nb_classes)的二值序列
              optimizer=RMSprop(),  # optimizer是优化器(优化参数的算法),SGD(随机梯度下降)
              metrics=['accuracy'])  # metrics 性能评估函数类似与目标函数, 只不过该性能的评估结果并不会用于训练.
# 当使用字符串形式指明accuracy和crossentropy时，keras会非常智能地确定应该使用metrics包下面的哪个函数。因为metrics包下的那些metric函数有不同的使用场景
from keras.callbacks import ModelCheckpoint
filepath = "model_acc"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
#print(checkpoint)
callbacks_list = [checkpoint]

history = model.fit(X_train, Y_train, batch_size=config.batch_size, epochs=config.nb_epoch, verbose=1,callbacks=callbacks_list,
                    validation_data=(X_test, Y_test))


score = model.evaluate(X_test, Y_test, verbose=1)

x = X_test[:2]
x = x.reshape(2, config.max_len)
t = model.predict(x, batch_size=1, verbose=1)
print(t)
print("答案：",Y_test[:2])

print('loss:', score[0])
print('Test accuracy:', score[1])

