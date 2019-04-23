from keras.datasets import mnist
from keras.utils import np_utils
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

np.random.seed(10)

(X_train, y_train), (X_test, y_test) = mnist.load_data()

#将features转换为四维矩阵
X_train4D = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32')
X_test4D = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32')

#标准化
X_train4D_nomarlize = X_train4D / 255
X_test4D_normalize = X_test4D / 255

#label以One-Hot Encoding进行转换
y_trainOneHot = np_utils.to_categorical(y_train)
y_testOneHot = np_utils.to_categorical(y_test)

#建立Sequential线性堆叠模型
model = Sequential()

#建立卷积层1
model.add(Conv2D(filters=16,
                 kernel_size=(5,5),
                 padding='same',
                 input_shape=(28,28,1),
                 activation='relu'))

#建立池化层1
model.add(MaxPooling2D(pool_size=(2, 2)))

#建立卷积层2
model.add(Conv2D(filters=36,
                 kernel_size=(5,5),
                 padding='same',
                 activation='relu'))

#建立池化层2，并加入Dropout避免过度拟合
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))

#建立神经网络（平坦层、隐藏层、输出层）
model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(10, activation='softmax'))

#查看模型摘要
print(model.summary())


#进行训练
model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])

#开始训练
train_history = model.fit(x=X_train4D_nomarlize,
                          y=y_trainOneHot, validation_split=0.2,
                          epochs=10, batch_size=300, verbose=2)



#画出准确率执行结果
show_train_history('acc', 'val_acc')


#画出误差执行结果
show_train_history('loss', 'val_loss')


#评估模型准确率
scores = model.evaluate(X_test4D_normalize, y_testOneHot)
print(scores[1])

#进行预测
prediction = model.predict_classes(X_test4D_normalize)
print(prediction[:10])

