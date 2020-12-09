import os
import time
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import backend
from tensorflow.keras import regularizers
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger


class Inception_v3:
    def __init__(self, class_list, img_width, img_height, batch_size) -> None:

        backend.clear_session()

        self.class_list = class_list
        self.img_width, self.img_height = img_width, img_height
        self.batch_size = batch_size

        self.model = None

        self.train_data = None
        self.validation_data = None
        self.num_train_data = None

        self.day_now = time.strftime('%Y%m%d', time.localtime(time.time()))
        self.checkpointer = None
        self.csv_logger = None
        self.history = None

    def generate_train_val_data(self, num_train_data, data_dir='train/'):
        self.num_train_data = num_train_data
        _datagen = image.ImageDataGenerator(
            rescale=1. / 255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            validation_split=0.2
        )
        self.train_data = _datagen.flow_from_directory(
            data_dir,
            target_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='training'
        )
        self.validation_data = _datagen.flow_from_directory(
            data_dir,
            target_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='validation'
        )

    def set_model(self):

        self.model = InceptionV3(weights='imagenet', include_top=False)
        x = self.model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.2)(x)
        pred = Dense(len(self.class_list),
                     kernel_regularizer=regularizers.l2(0.005),
                     activation='softmax'
                     )(x)

        self.model = Model(inputs=self.model.input, outputs=pred)
        self.model.compile(optimizer=SGD(lr=0.0001, momentum=0.9),
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])

    def train(self, epochs=10):
        train_samples = self.num_train_data * 0.8
        val_samples = self.num_train_data * 0.2

        self.checkpointer = ModelCheckpoint(
            filepath='models/food_classifier_checkpoint_{}.hdf5'.format(
                self.day_now),
            verbose=1, save_best_only=True)
        self.csv_logger = CSVLogger(
            'logs/training/history_model_{}.log'.format(self.day_now))

        self.history = self.model.fit_generator(self.train_data,
                                                steps_per_epoch=train_samples // self.batch_size,
                                                validation_data=self.validation_data,
                                                validation_steps=val_samples // self.batch_size,
                                                epochs=epochs,
                                                verbose=1,
                                                callbacks=[self.csv_logger, self.checkpointer])

        self.model.save(
            'models/food_classifier_model_{}.hdf5'.format(self.day_now))

    def evaluation(self, data_dir='test/'):
        if self.model is not None:
            test_datagen = image.ImageDataGenerator(rescale=1./255)
            test_generator = test_datagen.flow_from_directory(
                data_dir,
                target_size=(self.img_height, self.img_width),
                batch_size=32,
                class_mode='categorical')
            scores = self.model.evaluate_generator(
                test_generator,
                steps=5)
            print('Evaluation data: {}'.format(data_dir))
            print("%s: %.2f%%" % (self.model.metrics_names[1], scores[1]*100))
        else:
            print('Model not found... : load_model or train plz')

    def prediction(self, img_path, show=True):
        img = image.load_img(img_path, target_size=())
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img /= 255.

        if self.model is not None:
            pred = self.model.predict(img)
            index = np.argmax(pred)
            self.class_list.sort()
            pred_value = self.class_list[index]
            if show:
                plt.imshow(img[0])
                plt.axis('off')
                plt.title('prediction: {}'.format(pred_value))
                plt.show()
        else:
            print('Model not found... : load_model or train plz')

    def load_recent_model(self):
        model_path = 'models/checkpoint/'
        model_list = os.listdir(model_path)
        if model_list:
            h5_list = [file for file in model_list if file.endswith(".hdf5")]
            h5_list.sort()
            backend.clear_session()
            self.model = load_model(h5_list[-1], compile=False)
            print('Model loaded...: ', h5_list[-1])
            return 1
        else:
            print('Model not found... : train plz')
            return 0

    def show_accuracy(self):
        if self.history is not None:
            title = 'model_accuracy_{}'.format(self.day_now)
            plt.title(title)
            plt.plot(self.history.history['acc'])
            plt.plot(self.history.history['val_acc'])
            plt.ylabel('accuracy')
            plt.xlabel('epoch')
            plt.legend(['train_acc', 'val_acc'], loc='best')
            plt.show()
            plt.savefig('results/accuracy_model_{}.png'.format(self.day_now))
        else:
            print('Model not found... : load_model or train plz')

    def show_loss(self):
        if self.history is not None:
            title = 'model_loss_{}'.format(self.day_now)
            plt.title(title)
            plt.plot(self.history.history['loss'])
            plt.plot(self.history.history['val_loss'])
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train_loss', 'val_loss'], loc='best')
            plt.show()
            plt.savefig('results/loss_model_{}.png'.format(self.day_now))
        else:
            print('Model not found... : load_model or train plz')
