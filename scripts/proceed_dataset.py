from .black_parts import get_the_black_parts
from .white_parts import get_the_white_parts
from .colors import calc_black_peak, calc_white_peak

import os
import random
import numpy as np
import cv2


def proceed_image(image):
    white_image,white_num = get_the_white_parts(image)
    black_image,black_num = get_the_black_parts(image)
    black_peak = calc_black_peak(image)
    white_peak = calc_white_peak(image)

    return np.array([white_num,black_num,black_peak,white_peak])


def create_set_from_images(folder,target):

    dataset = []
    images = []

    for filename in os.listdir(folder):
        image = cv2.imread(os.path.join(folder,filename))
        if image is not None:
            images.append(image)

    for image in images:
        dataset.append(proceed_image(image))

    return dataset,[target]*len(dataset)


def get_true_training_set():
    return create_set_from_images('./dataset/training/pandas',1)


def get_false_training_set():
    return create_set_from_images('./dataset/training/not_pandas',0)


def get_true_testing_set():
    return create_set_from_images('./dataset/test/pandas',1)


def get_false_testing_set():
    return create_set_from_images('./dataset/test/not_pandas',0)


def get_training_data():
    panda_set,panda_target = get_true_training_set()
    not_panda_set,not_panda_target = get_false_training_set()

    tupple_list = list(zip(panda_set + not_panda_set,panda_target + not_panda_target))

    random.shuffle(tupple_list)

    return list(zip(*tupple_list))


def get_test_data():
    panda_set,panda_target = get_true_testing_set()
    not_panda_set,not_panda_target = get_false_testing_set()

    tupple_list = list(zip(panda_set + not_panda_set,panda_target + not_panda_target))

    random.shuffle(tupple_list)

    return list(zip(*tupple_list))



if __name__ == '__main__':
    test,target = get_training_data()
    
    for data,tar in zip(test,target):
        print(data,tar)