from collections import defaultdict
import os
from shutil import copy


def ready_data():
    if "food-101" in os.listdir():
        print("Dataset already exists")
        data_dir = 'food-101/'
    else:
        print("Downloading the data...")
        # TO DO: Download data & extract
        # import wget
        # dl_add = 'http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz'
        # print("Dataset Downloaded...")
        # wget.download(url=dl_add, out='.')
        # print("Dataset Extracted...")
        # import tarfile
        # tar = tarfile.open(r"food-101.tar.gz", "r:gz")
        # for tarinfo in tar:
        #     tar.extract(tarinfo, r'.')
        # print("Dataset Download Done...")
        data_dir = 'food-101/food-101/'
    if not os.path.exists(os.path.join('logs', 'training')):
        os.makedirs(os.path.join('logs', 'training'))
    if not os.path.exists(os.path.join('models', 'checkpoint')):
        os.makedirs(os.path.join('models', 'checkpoint'))
    if not os.path.exists('results'):
        os.makedirs('results')
    return data_dir


def get_data(file_path, dest):
    num_imgs = 0
    src = file_path + 'images'
    mata_path = file_path + 'meta/{}.txt'.format(dest)
    classes_imgs = defaultdict(list)

    with open(mata_path, 'r') as txt:
        paths = [read.strip() for read in txt.readlines()]
        for p in paths:
            food = p.split('/')
            classes_imgs[food[0]].append(food[1] + '.jpg')
        num_imgs += len(paths)

    if not os.path.exists(os.path.join(file_path, dest)):
        print("no existed data/test folder : ", os.path.join(file_path, dest))
        temp = defaultdict(list)
        os.makedirs(os.path.join(file_path, dest))
        for food in classes_imgs.keys():
            print("Copying images into... : ", food)
            if not os.path.exists(os.path.join(file_path, dest, food)):
                os.makedirs(os.path.join(file_path, dest, food))
            for i in classes_imgs[food]:
                copy(os.path.join(src, food, i),
                     os.path.join(file_path, dest, food, i))
                temp[food].append(
                    os.path.join(file_path, dest, food, i))
        classes_imgs = temp
    else:
        print("already existed data/test folder : ", os.path.join(file_path, dest))
        temp = defaultdict(list)
        for food in classes_imgs.keys():
            for i in classes_imgs[food]:
                temp[food].append(
                    os.path.join(file_path, dest, food, i))
        classes_imgs = temp
    
    

    return classes_imgs, num_imgs
