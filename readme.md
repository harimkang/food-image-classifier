# food-image-classifier

![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fharimkang/food-image-classifier) ![License](https://img.shields.io/github/license/harimkang/food-image-classifier?style=plastic) ![Stars](https://img.shields.io/github/stars/harimkang/food-image-classifier?style=social)

![fc_logo](https://user-images.githubusercontent.com/38045080/102370810-db65ed80-4000-11eb-8da6-b8314b5e8107.png)

The repository is a classifier project that allows you to easily categorize food photos using the CLI. Currently, a total of 101 types of food can be classified, and the model can be trained using a new dataset according to code commands.

- [O] Food Image Classification
- [O] Learning a new model
- [O] Assessing existing and new models
- [Need-to-check] Transfer learning for the current model

> We have organized data sets and codes with the goal of matching the type of food, and we have achieved the goal by learning about them. These simple projects can easily be applied to new research and ideas in the future.

## Table of Contents

- [Environment](#Environment)
- [Installation](#Installation)
- [Usage](#Usage)
- [Examples](#Examples)
- [Dataset](#Dataset)
- [Model](#Model)
- [TEST](#TEST)
- [TO-DO](#TO-DO)
- [Contributing](#contributing)
- [License](#License)

## Environment

### Dependencies

food-image-classifier was developed using the following library version:

- [Python3] - 3.8.3
- [Tensorflow] - 2.3.0
- [CUDA] - 10.1
- [Cudnn] - 7.6.5

and GPU enable Environment (Slow, but also in a CPU environment.)

## Installation

1. Clone food-image-classifier Repository

  ```sh
  $ git clone https://github.com/harimkang/food-image-classifier.git
  $ cd food-image-classifier
  ```

2. Configure virtual environment

```sh
$ python -m venv .venv
$ (windows) .venv\Scripts\activate
$ (Linux) source .venv/bin/activate
```

3. Install the dependencies.

```sh
$ pip install -r requirements.txt
```

## Usage

* CLI Model

```sh
$ python food_classifier.py [-m/--mode] [-d/--data_set] [-t/--test_data] \
                            [-b/ --batch_size] [-e/--epoch]
```

## Examples

* prediction (If you have trained models/checkpoints)
  1. Put the picture of the food you want to predict in the examples folder.
  2. Perform the command below.

    ```sh
    $ python food_classifire -m pred -t A.jpg B.jpg
    ```

  3. check result - If you predict the picture of bibimbap found on Google, it's as below.
  ![prediction_bibimbap](https://user-images.githubusercontent.com/38045080/102379572-55e73b00-400a-11eb-87de-3828e6b4a96a.png)

* training new model
  1. Perform the command below.

    ```sh
    $ python food_classifire -m train
    ```

* evaluate model (If you have trained models/checkpoints)
  1. Perform the command below.

    ```sh
    $ python food_classifire -m eval
    ```

* test model (evaluate + pred some examples)
  1. Perform the command below.

    ```sh
    $ python food_classifire -m test
    Dataset already exists
    --------------------TEST Mode--------------------
    Classes : 101
    Model loaded...:  food_classifier_checkpoint_20201210.hdf5
    Found 25250 images belonging to 101 classes.
    Evaluation data: food-101/test/
    accuracy: 83.75%
    [Model Prediction] applepie: apple_pie
    [Model Prediction] pizza: pizza
    [Model Prediction] samosa: samosa
    ```

## Dataset

### [Food-101](<http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz>)

  - We found that the data set was opened with a data set called food-101 in the [kaggle](https://www.kaggle.com/dansbecker/food-101/).

  - There are 101 food classes in total, and there are 1000 each.

  - The dataset is divided into trains (75750 imgs, 75%)/test (25250 imgs, 25%) in metadata.

  - We only used the train (75750 imgs) set for training. Instead, the train set was divided into train (60600 imgs, 80%)/validation (15150 imgs, 20%) sets for model training. The remaining test sets (25250 imgs) were used only to evaluate the trained model.

    | Total | Train | Validation | Test |
    | ------ | ------ | ------ | ------ |
    | 101000 imgs | 60600 imgs | 15150 imgs | 25250 imgs |
    | 100 % | 60 % | 15 % | 25 % |

## Model

The project was written based on the inception_v3 model provided by Keras.application and was trained.

| Model | Reference |
| ------ | ------ |
| Inception v3 | [inception_v3](https://www.tensorflow.org/api_docs/python/tf/keras/applications/inception_v3)|

The model we have trained is as follows.

[pretrained_model](https://drive.google.com/file/d/1Y-ofMxVDF33Q_pIoe5K5qpFrWE-Ijctn/view?usp=sharing)

Once you download and extract the model data above, you can place the models folder on root basis. The above models are 10 epochs trained based on batch_size=16.

| Accuracy | Loss | Val_Accuracy | Val_Loss |
| ------ | ------ | ------ | ------ |
| 69.4 % | 1.612 | 72.5 % | 1.467 |

You can improve the performance of the model through transfer learning.
Alternatively, you can train with more appropriate hyperparameters or train longer to get a good performance model.

## TEST

We wrote the test code together to check the normal code operation. It is written as unittest of Python and needs coverage program to check coverage.

```sh
$ coverage run -m unittest discover tests
$ coverage report -i
Name                  Stmts   Miss  Cover
-----------------------------------------
model.py                122     46    62%
set_data_env.py          63     29    54%
tests\test_model.py      12      0   100%
tests\test_train.py      41      0   100%
-----------------------------------------
TOTAL                   238     75    68%
```

![image](https://user-images.githubusercontent.com/38045080/102384950-9ba70200-4010-11eb-9650-e42f4358bf1d.png)

## TO-DO

- New model plug-in
- New Dataset plug-in
- Increasing test code coverage

## Contributing

Feel free to [open an Issue](https://github.com/harimkang/food-image-classifier/issues/new), if you think something needs to be changed. You are welcome to participate in development, instructions are available in our contribution guide (TBD).

Or, if you have any questions, you can ask them via [email](mailto:harimkang4422@gmail.com).

## License

----
Copyright © 2020 [Harim Kang](https://github.com/harimkang).

MIT
