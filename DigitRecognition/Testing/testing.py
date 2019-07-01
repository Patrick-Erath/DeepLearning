import os
import cv2
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

TEST_FOLDER = '/home/one/Desktop/Images'
IMG_SIZE = 28
test_cases = ['3_00.jpeg','3_02.jpeg', '1_02.jpeg']
test_answer = 1

# Trained model from main.py
trained_model = tf.keras.models.load_model('mnist_reader.model')

mnist_data = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist_data.load_data()
predictions = trained_model.predict(x_test) # sample test data

path = os.path.join(TEST_FOLDER, test_cases[1])
img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
#img_test = tf.keras.utils.normalize(img, axis=1)
img_inverted = 255 - np.asarray(img)
img_resize = cv2.resize(img_inverted, (IMG_SIZE, IMG_SIZE))  # Resize input image to fit trained images (28x28 px)

# Create 3D array
array = np.zeros((1, 28, 28))
for i in range(img_resize[0].size):
    for j in range(img_resize[0].size):
        if(img_resize[i][j] < 105):
            img_resize[i][j] = 0
        array[0][i][j] = img_resize[i][j]

for i in range(3):
    print('    ')

# Shows image
plt.imshow(img_resize, cmap=plt.cm.binary)
plt.show()


if x_test[0][0].size != img_resize[0].size:
    print('image width error')
elif x_test[0].size != img_resize.size:
    print('image length error')
else:
    print('image dimensions correct...')

try:
    prediction = trained_model.predict(array)
    print('Prediction made successfully')
    print('Predicted number: ' + str(np.argmax(prediction[0])))
except Exception as e:
    print(e)