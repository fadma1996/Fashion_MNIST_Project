from __future__ import print_function
import keras
from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import print_summary
from keras.optimizers import Adam
from keras.regularizers import l2
import os
import sys
import numpy as np
from fashion_dataset import load_dataset
import keras.models

import keras.models

batch_size = 64
num_classes = 10

model_name = sys.argv[1]

epochs = int(sys.argv[2])
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
print(x_train.shape)
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
print(x_train.shape)

print("Validation data", sys.argv[3])
(x_val, y_val) = load_dataset(sys.argv[3])

flip_mnist_data=True
if flip_mnist_data:
    print("Flipping...")
    x_train_flip = np.flip(x_train, axis=2)

    x_train = np.append(x_train, x_train_flip, axis=0)
    y_train = np.append(y_train, y_train)


add_extra_data=False
if add_extra_data:
    print("Adding", sys.argv[4])
    (x_extra, y_extra) = load_dataset(sys.argv[4])
    x_train = np.append(x_train, x_extra, axis=0)
    y_train = np.append(y_train, y_extra, axis=0)

print("Final x_train.shape", x_train.shape)
print("Final y_train.shape", y_train.shape)


input_shape = (28, 28, 1)


x_train = x_train.astype('float32')
x_train /= 255

x_test = x_test.astype('float32')
x_test /= 255


x_val = x_val.astype('float32')
x_val /= 255


y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


y_val = keras.utils.to_categorical(y_val, num_classes)


model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same', kernel_regularizer=l2(0.01), input_shape=input_shape))
model.add(Activation('relu'))
model.add(Conv2D(32, (5, 5), kernel_regularizer=l2(0.01)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=l2(0.01)))
model.add(Activation('relu'))
model.add(Conv2D(64, (5, 5), kernel_regularizer=l2(0.01)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))
opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

from keras.callbacks import ModelCheckpoint
filepath=model_name + "-chkpt-e{epoch:02d}-acc{val_acc:.2f}.h5"
modelSaver = ModelCheckpoint(filepath, verbose=1, monitor="val_acc", save_best_only=True)


from keras.callbacks import CSVLogger
csv_logger = CSVLogger(model_name + '-training.log')

from keras.callbacks import TensorBoard
tensorboard = TensorBoard(log_dir="logs/{}".format(model_name))

callbacks_list = [modelSaver, tensorboard, csv_logger]



model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_val, y_val),
              shuffle=True,
              callbacks=callbacks_list)

transfer_learning=True
if transfer_learning:
         model.fit(x_extra,y_extra,
		      batch_size=batch_size,
		      epochs=epochs,
		      validation_data=(x_val, y_val),
		      shuffle=True,
		      callbacks=callbacks_list)

model.save_weights(model_name + "-weights.hdf5")

model.save(model_name + ".h5")

scores = model.evaluate(x_test, y_test, verbose=1)
print('x_test Test loss:', scores[0])
print('x_test Test accuracy:', scores[1])

scores = model.evaluate(x_val, y_val, verbose=1)
print('x_val Validation loss:', scores[0])
print('x_val Validation accuracy:', scores[1])


