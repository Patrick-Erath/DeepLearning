import os
import cv2
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

TEST_FOLDER = '/home/one/Desktop/Images'
IMG_SIZE = 28
test_cases = ['1_02.jpeg']
test_answer = 1

# Trained model from main.py
trained_model = tf.keras.models.load_model('mnist_reader.model')


mnist_data = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist_data.load_data()
predictions = trained_model.predict(x_test) # sample test data

path = os.path.join(TEST_FOLDER, test_cases[0])
img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
img_test = tf.keras.utils.normalize(img, axis=1)
img_resize = cv2.resize(img_test, (IMG_SIZE, IMG_SIZE))  # Resize input image to fit trained images (28x28 px)

# TODO: Invert black to white

# Covert image to 3D tuple
test_arr = [[[0 for x in range(img_resize.size)] for y in range(img_resize.size)] for z in range(1)]

for i in range(img_resize[0].size):
    for j in range(img_resize[0].size):
        test_arr[0][i][j] = img_resize[i][j]


for i in range(3):
    print('    ')

# Shows image
plt.imshow(x_test[0], cmap=plt.cm.binary)
#plt.show()


if x_test[0][0].size != img_resize[0].size:
    print('image width error')
elif x_test[0].size != img_resize.size:
    print('image length error')
else:
    print('image dimensions correct...')

try:
    prediction = trained_model.predict(test_arr)
    print('Prediction made successfully')
    print('Predicted number: ' + str(np.argmax(prediction[0])))
except Exception as e:
    print(e)


# personal tests (handrawn numbers)
#for test_case in test_cases:
    #path = os.path.join(TEST_FOLDER, test_case)
    #cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    #img_array =
    #prediction = new_model.predict()

