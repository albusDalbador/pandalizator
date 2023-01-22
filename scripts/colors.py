from math import ceil
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import scipy as sp
from scipy import ndimage
import cv2
import numpy as np
from PIL import Image


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])


def calc_black_peak(panda):
    panda_gray = rgb2gray(panda)
    bins = ndimage.histogram(panda_gray,0,255,256)
    return round(np.sum(bins[0:50])/np.sum(bins[50:100]),5)


def calc_white_peak(panda):
    panda_gray = rgb2gray(panda)
    bins = ndimage.histogram(panda_gray,0,255,256)
    return round(np.sum(bins[205:])/np.sum(bins[155:205]),5)

#for test purposes
if __name__ == '__main__':
    first_panda = np.array(Image.open('../images/test_panda.jpg'))
    second_panda = np.array(Image.open('../images/big_panda.png'))
    third_panda = np.array(Image.open('../images/panda_brzuch.jpg'))
    panda_array = [first_panda,second_panda,third_panda]

    figure,histogram = plt.subplots(5,3)

    for i in range(len(panda_array)):
        panda = panda_array[i]
        panda_gray = rgb2gray(panda)

        histogram[0,i].imshow(panda)
        histogram[1,i].plot(ndimage.histogram(panda_gray,0,255,256))
        histogram[2,i].imshow(panda[:,:,0])
        histogram[3,i].imshow(panda[:,:,1])
        histogram[4,i].imshow(panda[:,:,2])
    plt.show()