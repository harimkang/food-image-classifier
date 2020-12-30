"""
# Reference
- https://www.kaggle.com/boopesh07/multiclass-food-classification-using-tensorflow
"""
import os
from shutil import copy
from collections import defaultdict


def check_env_dir():
    # Create required directory
    if not os.path.exists(os.path.join("logs", "training")):
        os.makedirs(os.path.join("logs", "training"))
    if not os.path.exists(os.path.join("models", "checkpoint")):
        os.makedirs(os.path.join("models", "checkpoint"))
    if not os.path.exists(os.path.join("results", "train")):
        os.makedirs(os.path.join("results", "train"))

    return 1


def ready_data(data="food-101"):
    """
    # It checks whether the food-101 dataset exists,
    # if not, downloads it and returns the folder location.
    """
    if data == "food-101":
        if "food-101" in os.listdir():
            print("Dataset already exists")
            if not os.path.exists(os.path.join("food-101", "food-101")):
                data_dir = "food-101/"
            else:
                data_dir = "food-101/food-101/"
        else:
            if not os.path.exists(r"food-101.tar.gz"):
                # TODO: Download data & extract (NOT Check)
                print("Downloading the data...")
                import wget

                dl_add = "http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz"
                print("Dataset Downloaded...")
                wget.download(url=dl_add, out=".")
            print("Dataset Extracted...")
            import tarfile

            tar = tarfile.open(r"food-101.tar.gz", "r:gz")
            for tarinfo in tar:
                tar.extract(tarinfo, r".")
            print("Dataset Download Done...")
            data_dir = "food-101/food-101/"
    else:
        # TODO: other dataset added
        data_dir = None

    return data_dir


def get_data(file_path, dest):
    """
    # [input parameters]
        - file_path: The root path of the dataset returned by the ready_data() function.
        - dest: 'train' or 'test' -> This is the choice of what metadata to read. It also creates that folder as a result.
    # Separate the existing dataset into train and test folders and copy the files.
    # food-101 train dataset : 75750 imgs, 101 classes
    # food-101 test dataset : 25250 imgs, 101 classes
    """
    num_imgs = 0
    src = file_path + "images"
    mata_path = file_path + "meta/{}.txt".format(dest)
    classes_imgs = defaultdict(list)

    # Create a dictionary with the dataset label as the key and the image file name as the value.
    with open(mata_path, "r") as txt:
        paths = [read.strip() for read in txt.readlines()]
        for p in paths:
            food = p.split("/")
            classes_imgs[food[0]].append(food[1] + ".jpg")
        num_imgs += len(paths)

    if not os.path.exists(os.path.join(file_path, dest)):
        # Copy files matching dest to each appropriate folder.
        print("No existed data/test folder : ", os.path.join(file_path, dest))
        temp = defaultdict(list)
        os.makedirs(os.path.join(file_path, dest))
        for food in classes_imgs.keys():
            print("Copying images into... : ", food)
            if not os.path.exists(os.path.join(file_path, dest, food)):
                os.makedirs(os.path.join(file_path, dest, food))
            for i in classes_imgs[food]:
                copy(os.path.join(src, food, i), os.path.join(file_path, dest, food, i))
                temp[food].append(os.path.join(file_path, dest, food, i))
        classes_imgs = temp
    else:
        print("already existed data/test folder : ", os.path.join(file_path, dest))
        temp = defaultdict(list)
        for food in classes_imgs.keys():
            for i in classes_imgs[food]:
                temp[food].append(os.path.join(file_path, dest, food, i))
        classes_imgs = temp

    return classes_imgs, num_imgs
