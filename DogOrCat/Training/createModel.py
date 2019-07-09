import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
from tqdm import tqdm


DATADIR  = '/mnt/hgfs/SharedItemsVM/kagglecatsanddogs_3367a/PetImages'
CATEGORIES = ["Dog", "Cat"]
IMG_SIZE = 100
EXCEPTIONS = 0

training_data = []

def view_data():
    try:
        for category in CATEGORIES:
            path = os.path.join(DATADIR, category)
            for img in os.listdir(path):
                print(img)
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                plt.imshow(new_array, cmap='gray')
                plt.show()
                break
            break
    except:
        print('Error')

    print(img_array)
    print(img_array.shape)


def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)
        for img in tqdm(os.listdir(path)):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                training_data.append([new_array, class_num])
            except Exception as e:
                EXCEPTIONS += 1
                pass

create_training_data()
#view_data()
print(len(training_data))
print(EXCEPTIONS)