import tensorflow as tf
sess = tf.Session()
from keras import backend as K
K.set_session(sess)
sess.run(tf.local_variables_initializer())
sess.run(tf.global_variables_initializer())
import numpy as np
import tensorflow as tf

def lr_schedule(epoch):
    lrate = 0.001
    if epoch > 75:
        lrate = 0.0005
    if epoch > 100:
        lrate = 0.0003
    return lrate

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# z-score
mean = np.mean(x_train, axis=(0, 1, 2, 3))
std = np.std(x_train, axis=(0, 1, 2, 3))
x_train = (x_train - mean) / (std + 1e-7)
x_test = (x_test - mean) / (std + 1e-7)

from tensorflow.python.keras.utils import  np_utils

num_classes = 10
y_train = np_utils.to_categorical(y_train, num_classes)
y_test = np_utils.to_categorical(y_test, num_classes)
regularizers=tf.keras.regularizers
weight_decay = 1e-4
model = tf.keras.models.Sequential()
model.add(
    tf.keras.layers.Conv2D(32, (3, 3), padding='same',input_shape=(32,32,3) , kernel_regularizer=tf.keras.regularizers.l2(weight_decay)))
model.add(tf.keras.layers.Activation('relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Conv2D(32, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(tf.keras.layers.Activation('relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Dropout(0.2))

model.add(tf.keras.layers.Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(tf.keras.layers.Activation('relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(tf.keras.layers.Activation('relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Dropout(0.3))

model.add(tf.keras.layers.Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(tf.keras.layers.Activation('relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(tf.keras.layers.Activation('relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Dropout(0.4))

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))

model.summary()

from tensorflow.python.keras.utils import Sequence, np_utils
import numpy as np
import cv2

class mygenerator(Sequence):
    def __init__(self, x_set, y_set, batch_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        # read your data here using the batch lists, batch_x and batch_y
        x = [filename for filename in batch_x]
        y = [filename for filename in batch_y]
        return np.array(x), np.array(y)

image_gen=iter(mygenerator(x_train,y_train, 4))

# training
batch_size = 64

opt_loss = tf.keras.losses.categorical_crossentropy

model.compile(loss=opt_loss, optimizer=tf.keras.optimizers.RMSprop(lr=0.0001), metrics=['accuracy'])
model.fit_generator(image_gen, \
                    steps_per_epoch=x_train.shape[0] // batch_size, epochs=1, \
                    verbose=1, validation_data=(x_test, y_test), callbacks=[tf.keras.callbacks.LearningRateScheduler(lr_schedule)])
# save to disk

keras_file = "keras_model.h5"
tf.keras.models.save_model(model, keras_file)
#
# # Convert to TensorFlow Lite model.
converter = tf.lite.TFLiteConverter.from_keras_model_file(keras_file)
tflite_model = converter.convert()
open("converted_model.tflite", "wb").write(tflite_model)


# testing
# scores = model.evaluate(x_test, y_test, batch_size=128, verbose=1)
# print('\nTest result: %.3f loss: %.3f' % (scores[1] * 100, scores[0]))