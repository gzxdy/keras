from keras.layers import *
from keras import Model
import config
# from keras.optimizers import SGD, Adam, RMSprop
# import pickle
# from keras.callbacks import ModelCheckpoint
from keras.layers.embeddings import Embedding

def lstm():


    input = Input(shape=(config.max_len,))
    embed = Embedding(config.dic_len + 1, 100, input_length=config.max_len, trainable=True,mask_zero=True)(input)
    #在该参数中，如trainable=False,则意味着不对该层权重进行更新；
    #mask_zero=True, 索引 0 就不能被用于词汇表中 （则input_dim 应该与 vocabulary + 1 大小相同）。

    ###
    embed = Dropout(0.3)(embed)
    #embed = BatchNormalization()(embed)
    repre = Bidirectional(LSTM(units=200, return_sequences=False))(embed)     #return_sequences = False只返回输出序列的最后一个time step的输出
    #  return_sequences = True返回整个序列,每一个time step都会输出，比如stack两层LSTM时候要这么设置。https://blog.csdn.net/hellocsz/article/details/88802521

    output = Dense(units=3, activation="softmax")(repre)
    model = Model(input=input, output=output)

    return model
