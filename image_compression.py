import numpy as np
from numpy.linalg import svd
import sys
import time
import matplotlib.pyplot as plt

#compression amount should be 0 and 1. Where 20% compression is 0.2 or compression_amount can equal amount of columns you want compressed. 
#therefor input a number between 1 and max_columns(32 for cifar) k = amount of columns of U used. 
def compress_images(images, compression_amount=0.2):
    
    #images should be of size (batch_size, 32, 32, 3)
    compressed_images = []

    total = images.shape[0]
    #this is the amount of columns of U that will be used in the images compression
    if compression_amount >= 1:
        k = compression_amount
    else:
        k = int((1 - compression_amount) * 32)
    begin_time = time.time()
    eta = 0

    for i, img in enumerate(images):
        red = img[:, :, 0]
        green = img[:, :, 1]
        blue = img[:, :, 2]

        ru, rs, rv = svd(red)
        gu, gs, gv = svd(green)
        bu, bs, bv = svd(blue)

        final_red = np.dot(ru[:, 0:k] * rs[0:k], rv[0:k]).reshape(32,32)
        final_green = np.dot(gu[:, 0:k] * gs[0:k], gv[0:k]).reshape(32,32)
        final_blue = np.dot(bu[:, 0:k] * bs[0:k], bv[0:k]).reshape(32,32)

        final_img = np.zeros((32,32,3))
        final_img[:, :, 0] = final_red
        final_img[:, :, 1] = final_green
        final_img[:, :, 2] = final_blue

        bar_len = 60
        filled_len = int(round(bar_len * i / float(total)))
        percents = round(100.0 * i / float(total), 1)
        if percents % 1.0 == 0 and percents != 0.0:
            cur_elapsed = time.time() - begin_time 
            cur_elapsed = cur_elapsed / percents
            eta = cur_elapsed * (100.0 - percents)
        bar = '=' * filled_len + '-' * (bar_len - filled_len)

        sys.stdout.write('Compressed Images %s/%s [%s] %s%%  ETA: %2ss\r' % (i + 1, total, bar, percents, int(eta)))
        sys.stdout.flush()
        if (i + 1) == total:
           print('Compressed Images %s/%s [%s] %s%%  ETA: %2ss\r' % (i + 1, total, bar, percents, int(eta)) + "\n")
                

        compressed_images.append(final_img.astype(np.int32))

    compressed_images = np.array(compressed_images).reshape(-1, 32, 32, 3)


    return compressed_images

def plot_graphs(training_data_control, training_data_other, compress_list):
    # plot accuracy over training time
    plt.clf()
    train_legend = []
    for i, train in enumerate(training_data_control):
        plt.plot(train.history['sparse_categorical_accuracy'])
        plt.plot(train.history['val_sparse_categorical_accuracy'])
        train_legend.append("Train Acc %s Columns" %(compress_list[i]))
        train_legend.append("Test Acc %s Columns" %(compress_list[i]))

    plt.title('Accuracy Over Time Control Model')
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.legend(train_legend)
    plt.savefig('figures/acc_over_time_control.png')
    plt.show()
    plt.clf()

    train_legend = []
    for i, train in enumerate(training_data_other):
        plt.plot(train.history['sparse_categorical_accuracy'])
        plt.plot(train.history['val_sparse_categorical_accuracy'])
        train_legend.append("Train Acc %s Columns" %(compress_list[i]))
        train_legend.append("Test Acc %s Columns" %(compress_list[i]))

    plt.title('Accuracy Over Time 2nd Model')
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.legend(train_legend)
    plt.savefig('figures/acc_over_time_2nd_model.png')
    plt.show()
    plt.clf()

    # plot loss over training time
    train_legend = []
    for i, train in enumerate(training_data_control):
        plt.plot(train.history['loss'])
        plt.plot(train.history['val_loss'])
        train_legend.append("Train Loss %s Columns" %(compress_list[i]))
        train_legend.append("Test Loss %s Columns" %(compress_list[i]))

    plt.title('Loss Over Time Control Model')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend(train_legend)
    plt.savefig('figures/loss_over_time_control.png')
    plt.show()
    plt.clf()

    train_legend = []
    for i, train in enumerate(training_data_other):
        plt.plot(train.history['loss'])
        plt.plot(train.history['val_loss'])
        train_legend.append("Train Loss %s Columns" %(compress_list[i]))
        train_legend.append("Test Loss %s Columns" %(compress_list[i]))

    plt.title('Loss Over Time 2nd Model')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend(train_legend)
    plt.savefig('figures/loss_over_time_2nd_model.png')
    plt.show()
    plt.clf()
    


