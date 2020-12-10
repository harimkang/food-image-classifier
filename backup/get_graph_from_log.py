import matplotlib.pyplot as plt
import time

meta_path = 'logs/training/history_model_20201210.log'

accuracy = []
loss = []
val_accuracy = []
val_loss = []
i=0
with open(meta_path, 'r') as txt:
    for read in txt.readlines():
        if i != 0:
            temp = read.split(',')
            accuracy.append(float(temp[1]))
            loss.append(float(temp[2]))
            val_accuracy.append(float(temp[3]))
            val_loss.append(float(temp[4]))
        i += 1


epoch = [i for i in range(10)]
# print(loss)
title = 'model_accuracy_{}'.format(time.strftime('%Y%m%d', time.localtime(time.time())))
plt.title(title)
plt.plot(epoch, accuracy)
plt.plot(epoch, val_accuracy)
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train_acc', 'val_acc'], loc='best')
plt.axis([0, 9, 0, 1])     # X, Y축의 범위: [xmin, xmax, ymin, ymax]
plt.show()

title2 = 'model_loss_{}'.format(time.strftime('%Y%m%d', time.localtime(time.time())))
plt.title(title2)
plt.plot(epoch, loss)
plt.plot(epoch, val_loss)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train_loss', 'val_loss'], loc='best')
plt.axis([0, 9, 0, 5])     # X, Y축의 범위: [xmin, xmax, ymin, ymax]
plt.show()