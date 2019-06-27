import tensorflow as tf
import numpy as np

mnist_data = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist_data.load_data()

# normalize data -> scale between 0 and 1
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=3)

val_loss, val_acc = model.evaluate(x_test, y_test)
print(val_loss)
print(val_acc)

model.save('mnist_reader.model')
new_model = tf.keras.models.load_model('mnist_reader.model')
predictions = new_model.predict(x_test)

passed = 0
failed = 0
for i in range(len(predictions)):
    if(np.argmax(predictions[i]) == y_test[i]):
        print('Correctly predicted: ' + str(y_test[i]))
        passed += 1
    else:
        print('Falsely predicted: ' + str(np.argmax(predictions[i])) + ' Correct answer: ' + str(y_test[i]))
        failed += 1

print("Passed count " + str(passed))
print("Failed count " + str(failed))