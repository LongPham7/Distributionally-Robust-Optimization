import numpy as np

from keras.utils import to_categorical
from keras.datasets import mnist
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, Flatten, Activation
from keras.optimizers import Adam
from keras import backend as K

import foolbox
from foolbox.models import KerasModel
from foolbox.criteria import Misclassification

"""
This module is for sanity check. 
It creates the neural network used by Staib et al. and Sinha et al.
in Keras and evalutes its robustness (or vulnerability) against an
FGSM adversary.
"""

nb_filters = 64
epochs = 25
batch_size = 128
num_classes = 10

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
    channel_axis = 1
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)
    channel_axis = 3

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = to_categorical(y_train, num_classes)
y_test_original = y_test
y_test = to_categorical(y_test, num_classes)

def trainModel(activation='relu'):
    model = Sequential()
    model.add(Conv2D(filters=nb_filters, kernel_size=(8,8), strides=(2,2), padding='same', input_shape=input_shape))
    model.add(Activation(activation))
    model.add(Conv2D(filters=nb_filters * 2, kernel_size=(6,6), strides=(2,2), padding='valid'))
    model.add(Activation(activation))
    model.add(Conv2D(filters=nb_filters * 2, kernel_size=(5,5), strides=(1,1), padding='valid'))
    model.add(Activation(activation))
    model.add(Flatten())
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    optimizer = Adam(lr=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    filepath = './experiment_models/KerasMNISTClassifier_{}.h5'.format(activation)
    model.save(filepath)

def adversarialAccuracy(model):
    keras_model = KerasModel(model, bounds=(0,1), channel_axis=channel_axis)
    criterion = Misclassification()

    wrong, total = 0, x_test.shape[0]
    period = 500
    for i in range(total):
        image, label = x_test[i], y_test_original[i]

        attack = foolbox.attacks.FGSM(keras_model, criterion)
        image_adv = attack(image, label, epsilons=5, max_epsilon=1.0)
        if image_adv is not None:
            wrong += 1
        if i%period == period - 1:
            print("Adversarial attack success rate: {} / {} = {}".format(wrong, i+1, wrong / (i+1)))

    print("Adversarial error rate: {} / {} = {}".format(wrong, total, wrong / total))

if __name__ == "__main__":
    filepath_relu = './experiment_models/KerasMNISTClassifier_relu.h5'
    filepath_elu = './experiment_models/KerasMNISTClassifier_elu.h5'
    model_relu = load_model(filepath_relu)
    model_elu = load_model(filepath_elu)

    model_relu.summary()

    adversarialAccuracy(model_relu)
    adversarialAccuracy(model_elu)
