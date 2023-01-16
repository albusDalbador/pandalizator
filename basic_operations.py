import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import scipy as sp
from scipy import ndimage
import cv2
import numpy as np
from PIL import Image
from math import isclose

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

first_panda = np.array(Image.open('./images/test_panda.jpg'))/255
second_panda = np.array(Image.open('./images/big_panda.png'))/255
third_panda = np.array(Image.open('./images/panda_00111.jpg'))/255
panda_array = [first_panda,second_panda,third_panda]

gray_first_panda = np.array(Image.open('./images/test_panda.jpg').convert('L'))/255
gray_second_panda = np.array(Image.open('./images/big_panda.png').convert('L'))/255
gray_third_panda = np.array(Image.open('./images/panda_00111.jpg').convert('L'))/255
gray_panda_array = [gray_first_panda,gray_second_panda,gray_third_panda]

figure,axis = plt.subplots(3,3)

for i  in range(0,3):
    gray_panda = gray_panda_array[i]
    panda = panda_array[i]
    panda_gray = rgb2gray(panda)
    black_parts = np.where(panda_gray < 0.1, 1, 0)
    binary_closed_black = ndimage.binary_closing(black_parts,iterations=5)
    greyscale_closed = ndimage.grey_closing(panda_gray,size=(25,25))
    greyscale_black_extracted = np.where(greyscale_closed < 0.2, 1, 0)

    # close_filtered = np.where(isclose(panda[:,:,0],panda[:,:,1],rel_tol=0.01) & isclose(panda[:,:,0],panda[:,:,2],rel_tol=0.01) & isclose(panda[:,:,1],panda[:,:,2],rel_tol=0.01),panda[:,:,[255,255,255]],panda[:,:,[0,0,0]])
    close_filtered = np.zeros((panda.shape[0],panda.shape[1]))
    for x in range (panda.shape[0]):
        for y in range(panda.shape[1]):
            if isclose(panda[x,y,0],panda[x,y,1],rel_tol=0.03) and isclose(panda[x,y,0],panda[x,y,2],rel_tol=0.03) and isclose(panda[x,y,2],panda[x,y,1],rel_tol=0.03):
                close_filtered[x,y] = 1
            else:
                close_filtered[x,y] = 0

    axis[0,i].imshow(close_filtered,cmap='gray')
    axis[1,i].imshow(binary_closed_black,cmap='gray')
    axis[2,i].imshow(greyscale_closed,cmap='gray')
    


plt.show()




















# print(gray_panda.shape)
# plt.imshow(gray_panda,cmap='gray')

# filtered_panda = np.where( (gray_panda < 0.1) | (gray_panda > 0.9) , 1,0 )


# plt.figure()
# plt.imshow(filtered_panda,cmap='gray')


# average_panda = np.sum(panda,2)/3
# rgb_filtered_panda = np.where( (panda[2] < 0.1) | (average_panda > 0.9), 1, 0)

# rgb_filtered_panda = np.zeros((panda.shape[0],panda.shape[1]))
# for i in range(0,panda.shape[0]):
#     for j in range(0,panda.shape[1]):
#         rgb_filtered_panda[i][j] = 0 if panda[i][j][1] > panda[i][j][2] and panda[i][j][1] > panda[i][j][0] else 1


# print(rgb_filtered_panda.shape)
# plt.figure()
# plt.imshow(rgb_filtered_panda,cmap='gray')


# black_parts = np.where(gray_panda < 0.2, 1, 0)
# white_parts = np.where(gray_panda > 0.8,1,0)

# plt.figure()
# plt.imshow(black_parts,cmap='gray')


# erosed_black_parts = ndimage.binary_erosion(black_parts,iterations=10)
# erosed_white_parts = ndimage.binary_erosion(white_parts,iterations=10)
# dilated_black_parts = ndimage.binary_dilation(erosed_blak_parts,iterations=50)
# plt.figure()
# plt.imshow(erosed_blak_parts,cmap='gray')
# # plt.figure()
# # plt.imshow(dilated_black_parts,cmap='gray')
# gauss_filtered_panda = ndimage.gaussian_filter(black_parts,sigma=1)
# gauss_filtered_panda = np.where(gauss_filtered_panda > 0.9, 1, 0)
# plt.figure()
# plt.imshow(gauss_filtered_panda,cmap='gray')


# plt.figure()
# plt.imshow(white_parts)

# panda_shape = np.add(erosed_white_parts,erosed_black_parts)
# panda_shape_filtered = ndimage.gaussian_filter(panda_shape,sigma=2)
# panda_shape_eroded = ndimage.binary_erosion(panda_shape,iterations=5)
# panda_filtered_dilated = ndimage.binary_dilation(panda_shape_filtered,iterations=50)
# panda_eroded_dilated = ndimage.binary_dilation(panda_shape_eroded,iterations=50)
# panda_closed = ndimage.binary_closing(panda_shape_eroded,iterations=50)
# panda_black_closed = ndimage.binary_closing(black_parts,iterations=50)

# figure, axis = plt.subplots(3,3)
# axis[0,0].imshow(erosed_white_parts,cmap='gray')
# axis[1,0].imshow(erosed_black_parts,cmap='gray')
# axis[2,0].imshow(panda_shape,cmap='gray')
# axis[0,1].imshow(panda)
# axis[1,1].imshow(panda_shape_filtered,cmap='gray')
# axis[2,1].imshow(panda_filtered_dilated,cmap='gray')
# axis[0,2].imshow(panda_closed,cmap='gray')
# axis[1,2].imshow(panda_shape_eroded,cmap='gray')
# axis[2,2].imshow(panda_eroded_dilated,cmap='gray')
# axis[0,0].imshow(black_parts,cmap='gray')
# axis[1,0].imshow(panda_black_closed,cmap='gray')

# figure.canvas.manager.full_screen_toggle()
# plt.get_current_fig_manager.full_screen_toggle()