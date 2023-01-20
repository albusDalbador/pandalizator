from math import ceil
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import scipy as sp
from scipy import ndimage
import cv2
import numpy as np
from PIL import Image
from scipy.special import euler


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])


def get_the_white_parts(panda):
    panda_gray = rgb2gray(panda)
    white_parts = np.where(panda_gray > 200, 1.0, 0.0)
    step_closed = np.copy(white_parts)
    step_eroded = np.copy(white_parts)

    iterations = ceil(panda_gray.shape[0]/100)

    for _ in range(4):
        step_eroded = ndimage.binary_erosion(step_closed,iterations=2)
        step_closed = ndimage.binary_closing(step_eroded,iterations=iterations)

    final_closed = ndimage.binary_erosion(step_closed,iterations=10)
    final_closed = ndimage.binary_dilation(final_closed,iterations=iterations*2)
    final_closed = final_closed.astype(np.uint8)
    
    (contours,hierarchy) = cv2.findContours(final_closed,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    # print(len(contours))
    return final_closed,len(contours)


if __name__ == "__main__":

    first_panda = np.array(Image.open('./images/test_panda.jpg'))
    second_panda = np.array(Image.open('./images/big_panda.png'))
    third_panda = np.array(Image.open('./images/panda_brzuch.jpg'))
    panda_array = [first_panda,second_panda,third_panda]


    figure,axis = plt.subplots(4,3)

    for i in range(len(panda_array)):
        panda = panda_array[i]
        panda_gray = rgb2gray(panda)
        white_parts = np.where(panda_gray > 200, 1.0, 0.0)
        adaptive_binarized = cv2.adaptiveThreshold(panda_gray.astype(np.uint8),255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,101,2)
        step_closed = np.copy(white_parts)
        step_eroded = np.copy(white_parts)

        iterations = ceil(panda_gray.shape[0]/100)

        for _ in range(7):
            step_eroded = ndimage.binary_erosion(step_closed,iterations=2)
            step_closed = ndimage.binary_closing(step_eroded,iterations=iterations)

        final_closed = ndimage.binary_erosion(step_closed,iterations=10)
        final_closed = ndimage.binary_dilation(final_closed,iterations=iterations*2)
        final_closed = final_closed.astype(np.uint8)

        print(type(final_closed[0][0]))
        
        (contours,hierarchy) = cv2.findContours(final_closed,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
        print(len(contours))

        axis[0,i].imshow(adaptive_binarized,cmap='gray')
        axis[1,i].imshow(white_parts,cmap='gray')
        axis[2,i].imshow(step_closed,cmap='gray')
        axis[3,i].imshow(final_closed,cmap='gray')


    plt.show()