from unittest import TestCase

from set_data_env import ready_data, get_data, check_env_dir
from model import Inception_v3


class Food101Test(TestCase):
    def setUp(self) -> None:
        self.data_set = "food-101"
        self.img_width, self.img_height = 299, 299

        self.file_path = ready_data(self.data_set)

        class_list = "tests/assets/classes.txt"
        with open(class_list, "r") as txt:
            self.label = [read.strip() for read in txt.readlines()]

        self.model = Inception_v3(
            class_list=self.label,
            img_width=self.img_width,
            img_height=self.img_height,
            batch_size=16,
        )

    def test_ready_data_food_101(self):
        self.assertIsNotNone(self.file_path)

    def test_dir_env_check(self):
        result = check_env_dir()
        self.assert_(result)

    def test_get_data_train(self):
        train_data, num_train_data = get_data(self.file_path, "train")
        self.assert_(num_train_data)

    def test_get_data_test(self):
        test_data, num_test_data = get_data(self.file_path, "test")
        self.assert_(num_test_data)

    def test_model_generate_food_101(self):
        self.assertIsNotNone(self.model)

    def test_train_val_data_generate(self):
        self.model.generate_train_val_data(data_dir=self.file_path + "train/")

        self.assertIsNotNone(self.model.train_data)
        self.assertIsNotNone(self.model.validation_data)

    def test_set_model_food_101(self):
        self.model.generate_train_val_data(data_dir=self.file_path + "train/")
        result = self.model.set_model()

        self.assert_(result)

    def test_prediction_food_3_images(self):
        if self.model.load():
            # Model Prediction
            result1 = self.model.prediction(
                img_path="examples/applepie.jpg", show=False
            )
            self.assert_(result1)
            result2 = self.model.prediction(img_path="examples/pizza.jpg", show=False)
            self.assert_(result2)
            result3 = self.model.prediction(img_path="examples/samosa.jpg", show=False)
            self.assert_(result3)
