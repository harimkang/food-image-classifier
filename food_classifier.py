import argparse

from set_data_env import ready_data, get_data, check_env_dir
from model import Inception_v3


if __name__ == "__main__":
    # CLI
    parser = argparse.ArgumentParser(description='Argument Supported')
    parser.add_argument('-m', '--mode', default='test',
                        help='Select Mode - train or test or eval [default: test]')
    parser.add_argument('-d', '--data_set', default='food-101',
                        help='Select input path folder')
    parser.add_argument('-t', '--test_data', nargs='+',
                        default=None, help='input test imgs path')
    parser.add_argument('-b', '--batch_size', default=16,
                        help='input train batch size')
    parser.add_argument('-e', '--epoch', default=10, help='input train epoch')

    args = parser.parse_args()
    mode = args.mode.strip()
    data_set = args.data_set.strip()
    test_data_path = args.test_data
    batch_size = args.batch_size
    epoch = args.epoch

    # Configuring the dataset environment for training
    file_path = ready_data(data_set)
    check_env_dir()
    img_width, img_height = 299, 299

    print('--------------------{} Mode--------------------'.format(mode.upper()))
    if mode.lower() == 'train':
        train_data, num_train_data = get_data(file_path, 'train')
        test_data, num_test_data = get_data(file_path, 'test')

        label = list(train_data.keys())
        n_class = len(label)

        # Outputting configured data set metadata information
        print('Classes :', n_class)
        print('Train Data: ', int(num_train_data * 0.8))
        print('Validation Data: ', int(num_train_data * 0.2))
        print('Test Data: ', num_test_data)
        # Model
        """
        # tensorflow.python.framework.errors_impl.ResourceExhaustedError: OOM
        # --> batch_size reduce! GTX1650 (4GB) : 16 OK
        """
        model = Inception_v3(class_list=label, img_width=img_width, img_height=img_height,
                             batch_size=batch_size)

        model.generate_train_val_data(data_dir=file_path + 'train/')
        model.set_model()
        model.train(epochs=epoch)
        model.show_accuracy()
        model.show_loss()
        model.evaluation(batch_size=batch_size, data_dir=file_path + 'test/')

    elif mode.lower() in ['test', 'eval', 'pred']:
        class_list = file_path + 'meta/classes.txt'
        with open(class_list, 'r') as txt:
            label = [read.strip() for read in txt.readlines()]

        model = Inception_v3(class_list=label, img_width=img_width, img_height=img_height,
                             batch_size=batch_size)

        print('Classes :', len(label))

        if mode.lower() == 'test':
            if model.load():
                model.evaluation(batch_size=batch_size,
                                 data_dir='food-101/test/')
                # Model Prediction
                model.prediction(img_path='examples/applepie.jpg')
                model.prediction(img_path='examples/pizza.jpg')
                model.prediction(img_path='examples/samosa.jpg')
            else:
                # If there is no pre-trained model
                print(
                    'Test Mode : No model was found. Please proceed with the training first.')

        elif mode.lower() == 'eval':
            if model.load():
                model.evaluation(batch_size=batch_size,
                                 data_dir='food-101/test/')
            else:
                # If there is no pre-trained model
                print(
                    'Eval Mode : No model was found. Please proceed with the training first.')

        elif mode.lower() == 'pred':
            if model.load():
                if test_data_path:
                    for data_name in test_data_path:
                        model.prediction(
                            img_path='examples/{}'.format(data_name))
                else:
                    print(
                        'Pred Mode : Please put the name of the image to be predicted')
            else:
                # If there is no pre-trained model
                print(
                    'Pred Mode : No model was found. Please proceed with the training first.')
