"""
# Reference
- https://www.kaggle.com/boopesh07/multiclass-food-classification-using-tensorflow
"""
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # tensorflow logging off
import time
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import regularizers
from tensorflow.keras import backend


class Inception_v3:
    """
    [Inception V3 Model]
    # Class created using inception_v3 provided in keras.applications
    """

    def __init__(self, class_list, img_width, img_height, batch_size) -> None:

        backend.clear_session()

        self.class_list = class_list
        self.img_width, self.img_height = img_width, img_height
        # batch_size can be up to 16 based on GPU 4GB (not available for 32)
        self.batch_size = batch_size

        self.model = None

        self.train_data = None
        self.validation_data = None
        self.num_train_data = None

        self.day_now = time.strftime("%Y%m%d%H%M", time.localtime(time.time()))
        self.checkpointer = None
        self.csv_logger = None
        self.history = None

    def generate_train_val_data(self, data_dir="train/"):
        """
        # Create an ImageDataGenerator by dividing the train and validation set
        # by 0.8/0.2 based on the train dataset folder.
        # train : 60600 imgs / validation : 15150 imgs
        """
        num_data = 0
        for root, dirs, files in os.walk(data_dir):
            if files:
                num_data += len(files)

        self.num_train_data = num_data
        _datagen = image.ImageDataGenerator(
            rescale=1.0 / 255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            validation_split=0.2,
        )
        self.train_data = _datagen.flow_from_directory(
            data_dir,
            target_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
            class_mode="categorical",
            subset="training",
        )
        self.validation_data = _datagen.flow_from_directory(
            data_dir,
            target_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
            class_mode="categorical",
            subset="validation",
        )

    def set_model(self):
        """
        # This is a function that composes a model, and proceeds to compile.
        # [Reference] - https://www.tensorflow.org/api_docs/python/tf/keras/applications/inception_v3
        """
        self.model = InceptionV3(weights="imagenet", include_top=False)
        x = self.model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(128, activation="relu")(x)
        x = Dropout(0.2)(x)
        pred = Dense(
            len(self.class_list),
            kernel_regularizer=regularizers.l2(0.005),
            activation="softmax",
        )(x)

        self.model = Model(inputs=self.model.input, outputs=pred)
        self.model.compile(
            optimizer=SGD(lr=0.0001, momentum=0.9),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        return 1

    def train(self, epochs=10):
        """
        # Training-related environment settings (log, checkpoint) and training
        """
        train_samples = self.num_train_data * 0.8
        val_samples = self.num_train_data * 0.2

        self.checkpointer = ModelCheckpoint(
            filepath="models/food_classifier_checkpoint_{}.hdf5".format(self.day_now),
            verbose=1,
            save_best_only=True,
        )
        self.csv_logger = CSVLogger(
            "logs/training/history_model_{}.log".format(self.day_now)
        )

        self.history = self.model.fit_generator(
            self.train_data,
            steps_per_epoch=train_samples // self.batch_size,
            validation_data=self.validation_data,
            validation_steps=val_samples // self.batch_size,
            epochs=epochs,
            verbose=1,
            callbacks=[self.csv_logger, self.checkpointer],
        )

        self.model.save("models/food_classifier_model_{}.hdf5".format(self.day_now))

    def evaluation(self, batch_size=16, data_dir="test/", steps=5):
        """
        # Evaluate the model using the data in data_dir as a test set.
        """
        if self.model is not None:
            test_datagen = image.ImageDataGenerator(rescale=1.0 / 255)
            test_generator = test_datagen.flow_from_directory(
                data_dir,
                target_size=(self.img_height, self.img_width),
                batch_size=batch_size,
                class_mode="categorical",
            )
            scores = self.model.evaluate_generator(test_generator, steps=steps)
            print("Evaluation data: {}".format(data_dir))
            print("%s: %.2f%%" % (self.model.metrics_names[1], scores[1] * 100))
        else:
            print("Model not found... : load_model or train plz")

    def prediction(self, img_path, show=True, save=False):
        """
        # Given a path for an image, the image is predicted and displayed through plt.
        """
        target_name = img_path.split(".")[0]
        target_name = target_name.split("/")[-1]
        save_time = time.strftime("%Y%m%d%H%M%S", time.localtime(time.time()))

        img = image.load_img(img_path, target_size=(self.img_height, self.img_width))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img /= 255.0

        if self.model is not None:
            pred = self.model.predict(img)
            index = np.argmax(pred)
            self.class_list.sort()
            pred_value = self.class_list[index]
            if show:
                plt.imshow(img[0])
                plt.axis("off")
                plt.title("prediction: {}".format(pred_value))
                print("[Model Prediction] {}: {}".format(target_name, pred_value))
                plt.show()
                if save:
                    plt.savefig(
                        "results/example_{}_{}.png".format(target_name, save_time)
                    )
            return 1
        else:
            print("Model not found... : load_model or train plz")
            return 0

    def load(self):
        """
        # If an already trained model exists, load it.
        """
        model_path = "models/checkpoint/"
        model_list = os.listdir(model_path)
        if model_list:
            h5_list = [file for file in model_list if file.endswith(".hdf5")]
            h5_list.sort()
            backend.clear_session()
            self.model = load_model(model_path + h5_list[-1], compile=False)
            print("Model loaded...: ", h5_list[-1])
            self.model.compile(
                optimizer=SGD(lr=0.0001, momentum=0.9),
                loss="categorical_crossentropy",
                metrics=["accuracy"],
            )
            return 1
        else:
            print("Model not found... : train plz")
            return 0

    def show_accuracy(self):
        """
        # Shows the accuracy graph of the training history.
        # TO DO: In the case of a loaded model, a function to find and display the graph is added
        """
        if self.history is not None:
            title = "model_accuracy_{}".format(self.day_now)
            plt.title(title)
            plt.plot(self.history.history["accuracy"])
            plt.plot(self.history.history["val_accuracy"])
            plt.ylabel("accuracy")
            plt.xlabel("epoch")
            plt.legend(["train_acc", "val_acc"], loc="best")
            plt.show()
            plt.savefig("results/accuracy_model_{}.png".format(self.day_now))
        else:
            print("Model not found... : load_model or train plz")

    def show_loss(self):
        """
        # Shows the loss graph of the training history.
        # TO DO: In the case of a loaded model, a function to find and display the graph is added
        """
        if self.history is not None:
            title = "model_loss_{}".format(self.day_now)
            plt.title(title)
            plt.plot(self.history.history["loss"])
            plt.plot(self.history.history["val_loss"])
            plt.ylabel("loss")
            plt.xlabel("epoch")
            plt.legend(["train_loss", "val_loss"], loc="best")
            plt.show()
            plt.savefig("results/loss_model_{}.png".format(self.day_now))
        else:
            print("Model not found... : load_model or train plz")
