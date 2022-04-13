import numpy as np
from numpy.linalg import svd
import sys


#compression amount should be 0 and 1. Where 20% compression is 0.2
def compress_images(images, compression_amount=0.2):
    
    #images should be of size (batch_size, 32, 32, 3)
    compressed_images = []

    total = images.shape[0]
    #this is the amount of columns of U that will be used in the images compression
    k = int((1 - compression_amount) * 32)

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
        bar = '=' * filled_len + '-' * (bar_len - filled_len)
        sys.stdout.write('Compressed Images %s/%s [%s] %s%%\r' % (i + 1, total, bar, percents))
        #if (i + 1) != total:
        sys.stdout.flush()
        if (i + 1) == total:
           print('Compressed Images %s/%s [%s] %s%%\r' % (i + 1, total, bar, percents) + "\n")
                
        
            

        #if (i+1) % 10000 == 0:
            #print(f"Compressed {i+1} images")

        compressed_images.append(final_img.astype(np.int32))

    compressed_images = np.array(compressed_images).reshape(-1, 32, 32, 3)

    return compressed_images



