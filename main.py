from set_data_env import ready_data, get_data
from model import Inception_v3

# Check if GPU is enabled
# print(tf.__version__)
# print(tf.test.gpu_device_name())


if __name__ == "__main__":

    file_path = ready_data()
    train_data, num_train_data = get_data(file_path, 'train')
    test_data, num_test_data = get_data(file_path, 'test')

    label = list(train_data.keys())
    img_width, img_height = 299, 299
    n_class = len(label)

    print('Classes :', n_class)
    print('Train Data: ', int(num_train_data * 0.8))
    print('Validation Data: ', int(num_train_data * 0.2))
    print('Test Data: ', num_test_data)

    # Model
    model = Inception_v3(class_list=label, img_width=img_width, img_height=img_height,
                         batch_size=64)

    if model.load_recent_model():
        # Load Model..
        pass
    else:
        # if Not Found trained Model in models/checkpoint
        model.generate_train_val_data(
            num_train_data=num_train_data, data_dir='food-101/train/')
        model.set_model()
        model.train(epochs=10)
        model.show_accuracy()
        model.show_loss()
        model.evaluation(data_dir='food-101/test/')

    # Prediction
    model.prediction(img_path='examples/applepie.jpg')
    model.prediction(img_path='examples/pizza.jpg')
    model.prediction(img_path='examples/samosa.jpg')
