from __future__ import print_function
import keras
from keras.utils import print_summary

from keras.models import load_model
from keras.optimizers import Adam

import os
import sys

import numpy as np

from fashion_dataset import load_dataset

#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"] = ""

#---------------------------------
batch_size = 64 # You can try 64 or 128 if you'd like to
num_classes = 10
input_shape = (28, 28, 1)

model_name = sys.argv[1]

epochs = int(sys.argv[2])

# This is training (( new extra )) data
train_dataset_name=sys.argv[3]
(x_train, y_train) = load_dataset(train_dataset_name)

# This is real/new validation data
(x_val, y_val) = load_dataset(sys.argv[4])
print("x_val!!!!!!",x_val.shape,"y_val!!!!!!",y_val.shape)
y_train = keras.utils.to_categorical(y_train, num_classes)
x_train = x_train.astype('float32')
x_train /= 255

y_val = keras.utils.to_categorical(y_val, num_classes)
x_val = x_val.astype('float32')
x_val /= 255

short_name,_ = os.path.splitext(model_name)
short_train_dataset_name,_ = os.path.splitext(os.path.basename(train_dataset_name))
short_name = short_name + "_" + short_train_dataset_name
print("short name:", short_name)

from keras.callbacks import ModelCheckpoint
filepath=short_name + "-step2-chkpt-e{epoch:02d}-acc{val_acc:.2f}.h5"
modelSaver = ModelCheckpoint(filepath, verbose=1, monitor="val_acc", save_best_only=True)

# save training log
from keras.callbacks import CSVLogger
csv_logger = CSVLogger(short_name + '-step2-training.log')

from keras.callbacks import TensorBoard
tensorboard = TensorBoard(log_dir="logs/{}-step2".format(short_name))

callbacks_list = [modelSaver, tensorboard, csv_logger]
#---------------load model-----------------------------:
model = load_model(model_name)

scores = model.evaluate(x_val, y_val, verbose=1)
print('Initial validation loss:', scores[0])
print('Initial validation accuracy:', scores[1])

opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

print(x_train.shape)
print(y_train.shape)

model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_val, y_val),
              shuffle=True,
              callbacks=callbacks_list)

model.save_weights(short_name + "-step2-weights.hdf5")

model.save(short_name + "-step2.h5")

scores = model.evaluate(x_val, y_val, verbose=1)
print('Step2 validation loss:', scores[0])
print('Step2 validation accuracy:', scores[1])


