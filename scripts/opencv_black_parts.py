from pickletools import uint8
import numpy as np 
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from scipy import ndimage
from math import floor,ceil

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

# first_panda = np.array(Image.open('./images/test_panda.jpg'))
# second_panda = np.array(Image.open('./images/big_panda.png'))
# third_panda = np.array(Image.open('./images/panda_00111.jpg'))

first_panda = cv2.imread('./images/test_panda.jpg',cv2.IMREAD_COLOR)
second_panda = cv2.imread('./images/big_panda.png',cv2.IMREAD_COLOR)
third_panda = cv2.imread('./images/panda_00111.jpg',cv2.IMREAD_COLOR)
panda_array = [first_panda,second_panda,third_panda]

figure,axis = plt.subplots(3,3)

for i in range(len(panda_array)):
    # medianed_panda = cv2.medianBlur(panda_array[i],5)
    gray_panda = cv2.cvtColor(panda_array[i],cv2.COLOR_BGR2GRAY)
    gray_panda_contours = cv2.cvtColor(panda_array[i],cv2.COLOR_BGR2GRAY)
    adaptive_binarized = cv2.adaptiveThreshold(gray_panda,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,101,2)
    _,usual_binarized = cv2.threshold(gray_panda,25,255,cv2.THRESH_BINARY)
    black_parts = np.where(gray_panda < 75, 0, 255)

    usual_binarized = cv2.bitwise_not(usual_binarized)

    # step_close = np.copy(usual_binarized)
    # step_open = np.copy(usual_binarized)
    # step_eroded = np.copy(usual_binarized)
    kernel_size = ceil(usual_binarized.shape[0]/250)
    kernel = np.ones((kernel_size,kernel_size),np.uint8)
    # erode_kernel = np.ones((4,4),np.uint8)
    # close_kernel = np.ones((7,7),np.uint8)

    closed = np.copy(usual_binarized)
    opened = np.copy(usual_binarized)

    for _ in range(3):
        opened = cv2.morphologyEx(closed,cv2.MORPH_OPEN,kernel,iterations=4)
        closed = cv2.morphologyEx(opened,cv2.MORPH_CLOSE,kernel,iterations=20)

    dst = cv2.fastNlMeansDenoising( closed, None, 15, 7, 21 )  
    print(type(closed[0][0]))
    # print(closed.shape)

    (contours,hierarchy) = cv2.findContours(closed,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    # print(len(contours))

    axis[0,i].imshow(usual_binarized,cmap='gray')
    # axis[1,i].imshow(opened,cmap='gray')
    axis[1,i].imshow(closed,cmap='gray')
    axis[2,i].imshow(dst,cmap='gray')
    # axis[3,i].imshow(panda_array[i],cmap='gray')

plt.show()