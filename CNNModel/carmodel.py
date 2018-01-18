from __future__ import print_function

import logging
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import ZeroPadding2D
from preprocess import ReadImages
from keras.layers.normalization import BatchNormalization


class Model:
    def _create_model(self, num_classes, input_shape):
        # Create the model
        model = Sequential()
        model.add(Conv2D(96, (11, 11), strides=4, input_shape=input_shape))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(5, 5), dim_ordering="th"))
        model.add(ZeroPadding2D((2, 2)))

        model.add(Conv2D(256, (5, 5)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(3, 3), dim_ordering="th"))
        model.add(ZeroPadding2D((1, 1)))

        model.add(Conv2D(384, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(3, 3), dim_ordering="th"))

        model.add(Conv2D(384, (3, 3)))
        model.add(Activation('relu'))

        model.add(Conv2D(256, (3, 3)))
        model.add(Activation('relu'))

        model.add(Flatten())
        model.add(Dense(1024))
        model.add(Activation('relu'))
        model.add(Dense(196))
        model.add(Activation('softmax'))
        # Compile model
        epochs = 10
        l_rate = 0.02
        #sgd = optimizers.SGD(lr=l_rate, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        print(model.summary())
        model.save('models/my_model.h5')
        return model, epochs

    def preprocess_and_build_model(self):
        logging.basicConfig(filename='log/info.log', level=logging.DEBUG, format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
        image_process = ReadImages()
        num_classes = 196

        # The data, shuffled and split between train and test sets:
        (X_train, y_train), (X_test, y_test) = image_process.get_train_and_test_data()
        logging.info('x_train shape: %s', str(X_train.shape))
        logging.info('train samples %s', str(X_train.shape[0]))
        logging.info('test samples %s', str(X_test.shape[0]))

        # normalize inputs from 0-255 to 0.0-1.0
        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')
        X_train = X_train / 255.
        X_test = X_test / 255.
        logging.info(X_train[0].shape)
        input_shape = X_train.shape[1:]
        # create our CNN model
        model, epochs = self._create_model(num_classes, input_shape)

        logging.info("CNN Model created.")

        # fit and run our model
        model.fit(X_train, y_train, nb_epoch=epochs, batch_size=40)
        # Final evaluation of the model
        scores = model.evaluate(X_test, y_test, verbose=0)
        logging.info("Accuracy: %.2f%%" % (scores[1] * 100))

        logging.info("done")