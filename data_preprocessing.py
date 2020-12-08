from collections import defaultdict
import os

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
    return data_dir
    

def get_data(file_path):
    num_imgs = 0
    classes_imgs = defaultdict(list)
    with open(file_path, 'r') as txt:
        paths = [read.strip() for read in txt.readlines()]
        for p in paths:
            food = p.split('/')
            classes_imgs[food[0]].append(food[1] + '.jpg')
        num_imgs += len(paths)
    return classes_imgs, num_imgs
