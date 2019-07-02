import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
from tqdm import tqdm
import random
import pickle
import time

class TrainModel():
    def __init__(self):
        self.DATADIR  = '/mnt/hgfs/SharedItemsVM/kagglecatsanddogs_3367a/PetImages'
        self.CATEGORIES = ["Dog", "Cat"]
        self.IMG_SIZE = 100
        self.EXCEPTIONS = 0
        self.X = []
        self.y = []
        self.training_data = []

    def view_data(self):
        try:
            for category in self.CATEGORIES:
                path = os.path.join(self.DATADIR, category)
                for img in os.listdir(path):
                    print(img)
                    img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                    new_array = cv2.resize(img_array, (self.IMG_SIZE, self.IMG_SIZE))
                    plt.imshow(new_array, cmap='gray')
                    plt.show()
                    break
                break
        except:
            print('Error')

        print(img_array)
        print(img_array.shape)


    def create_training_data(self):
        for category in self.CATEGORIES:
            path = os.path.join(self.DATADIR, category)
            class_num = self.CATEGORIES.index(category)
            for img in tqdm(os.listdir(path)):
                try:
                    img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                    new_array = cv2.resize(img_array, (self.IMG_SIZE, self.IMG_SIZE))
                    self.training_data.append([new_array, class_num])
                except Exception as e:
                    self.EXCEPTIONS += 1
                    pass
        random.shuffle(self.training_data)
        for features, label in self.training_data:
            self.X.append(features)
            self.y.append(label)
        self.X = np.array(self.X).reshape(-1, self.IMG_SIZE, self.IMG_SIZE, 1)
        #return self.X, self.Y, self.training_data

    def save_data(self):
        pickle_out = open('X.pickle', 'wb')
        pickle.dump(self.X, pickle_out)
        pickle_out.close()
        pickle_out = open('y.pickle', 'wb')
        pickle.dump(self.y, pickle_out)
        pickle_out.close()


train_obj = TrainModel()
t0 = time.time()
#train_obj.view_data()
train_obj.create_training_data()
train_obj.save_data()
time_delay = print(time.time() - t0)