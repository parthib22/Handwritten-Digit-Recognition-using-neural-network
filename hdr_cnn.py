import os
import cv2
import numpy as np
import tensorflow as tf

# Load and preprocess the MNIST dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape((60000, 28, 28, 1)).astype("float32") / 255
x_test = x_test.reshape((10000, 28, 28, 1)).astype("float32") / 255

y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)

# Build the CNN model
model = tf.keras.models.Sequential()
model.add(
    tf.keras.layers.Conv2D(64, (3, 3), activation="relu", input_shape=(28, 28, 1))
)
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation="relu"))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation="relu"))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(64, activation="relu"))
model.add(tf.keras.layers.Dense(32, activation="relu"))
model.add(tf.keras.layers.Dense(10, activation="softmax"))

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

model.fit(x_train, y_train, epochs=5)

model.save("handwritten_cnn.model")

# start here next time
model = tf.keras.models.load_model("handwritten_cnn.model")

image_number = 0
count = 0
datas = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
files = os.listdir("digits")
while os.path.isfile(f"digits/{image_number}.png"):
    try:
        img = cv2.imread(f"digits/{image_number}.png")[:, :, 0]
        img = np.invert(np.array([img]))
        img = img.reshape((1, 28, 28, 1)).astype("float32") / 255  # Reshape for CNN
        prediction = model.predict(img)
        print(
            f"A handwritten {datas[image_number]} is probably a {np.argmax(prediction)}"
        )
    except:
        print("Error!")
    finally:
        if np.argmax(prediction) == datas[image_number]:
            count += 1
        image_number += 1

print(f"Manual test model accuracy: {int(count/len(files)*100)}%")
