import numpy as np 
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from scipy import ndimage

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

# first_panda = np.array(Image.open('./images/test_panda.jpg'))
# second_panda = np.array(Image.open('./images/big_panda.png'))
# third_panda = np.array(Image.open('./images/panda_00111.jpg'))

first_panda = cv2.imread('./images/test_panda.jpg',cv2.IMREAD_COLOR)
second_panda = cv2.imread('./images/big_panda.png',cv2.IMREAD_COLOR)
third_panda = cv2.imread('./images/panda_00111.jpg',cv2.IMREAD_COLOR)
panda_array = [first_panda,second_panda,third_panda]

figure,axis = plt.subplots(4,3)

for i in range(len(panda_array)):
    # medianed_panda = cv2.medianBlur(panda_array[i],5)
    gray_panda = cv2.cvtColor(panda_array[i],cv2.COLOR_BGR2GRAY)
    gray_panda_contours = cv2.cvtColor(panda_array[i],cv2.COLOR_BGR2GRAY)
    adaptive_binarized = cv2.adaptiveThreshold(gray_panda,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,101,2)
    _,usual_binarized = cv2.threshold(gray_panda,25,255,cv2.THRESH_BINARY)
    black_parts = np.where(gray_panda < 75, 0, 255)

    minRadius = int(gray_panda.shape[1]/25)
    maxRadius = int(gray_panda.shape[1]/10)
    minDist = int(gray_panda.shape[1]/8)

    circles = cv2.HoughCircles(usual_binarized,cv2.HOUGH_GRADIENT,dp=1,minDist=minDist,param1=100,param2=15,minRadius=minRadius,maxRadius=maxRadius)
    circles = np.uint16(np.around(circles))
    print(circles.shape)
    for data in circles[0,:]:
        cv2.circle(gray_panda,(data[0],data[1]),data[2],(255,0,0),3)


    # contours,hierarchy = cv2.findContours(gray_panda_contours,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    # cv2.drawContours(gray_panda_contours,contours,contourIdx=-1,color=(0,255,0),thickness=3)
    edged = cv2.Canny(gray_panda_contours,25,255)
    contours, hierarchy = cv2.findContours(edged,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(panda_array[i],contours,-1,(0,255,0),3)
        
    axis[0,i].imshow(usual_binarized,cmap='gray')
    axis[1,i].imshow(adaptive_binarized,cmap='gray')
    axis[2,i].imshow(gray_panda,cmap='gray')
    axis[3,i].imshow(panda_array[i],cmap='gray')

plt.show()