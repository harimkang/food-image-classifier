from data_preprocessing import ready_data, get_data
from inception_v4 import Inception_v4

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Check if GPU is enabled
# print(tf.__version__)
# print(tf.test.gpu_device_name())




def train(input_shape, num_classes):
    
    batch_size = 16

    # train data_set
    
    # test data_set

    # get_model
    model = Inception_v4(input_shape, classes=num_classes)

if __name__ == "__main__":

    file_path = ready_data()
    train_data_path, num_train_data = get_data(file_path + 'meta/train.txt')
    test_data, num_test_data = get_data(file_path + 'meta/test.txt')
    label = list(test_data.keys())

    img_ch, img_width, img_height = 3, 299, 299
    n_class = len(label)

    train((img_ch, img_width, img_height), n_class)
    print(len(label))
    print(num_test_data)