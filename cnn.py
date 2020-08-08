import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import pickle

import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm



pickle_in = open("X.pickle","rb")
X = pickle.load(pickle_in)

pickle_in = open("y.pickle","rb")
y = pickle.load(pickle_in)

X = X/255.0

def create_model():
  model = tf.keras.models.Sequential([
    keras.layers.Conv2D(filters=256, kernel_size=(3, 3),activation='relu', input_shape=X.shape[1:]),
    keras.layers.MaxPool2D(pool_size=(2,2)),

    keras.layers.Conv2D(filters=256, kernel_size=(3, 3),activation='relu'),
    keras.layers.MaxPool2D(pool_size=(2,2)),

    keras.layers.Flatten(),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')

  ])

  model.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy'])

  return model

# model = create_model()

# model.summary()

IMG_SIZE = 50

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

y = np.array(y)


checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# # Create a callback that saves the model's weights
# cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
#                                                  save_weights_only=True,
#                                                  verbose=1)

# Train the model with the new callback
# model.fit(X, 
#           y,  
#           batch_size=32,
#           epochs=3,
#           validation_data=(X,y),
#           callbacks=[cp_callback])  # Pass callback to training



model1 = create_model()

# Evaluate the model
loss, acc = model1.evaluate(X,  y, verbose=2)
print("Untrained model, accuracy: {:5.2f}%".format(100*acc))
var = model1.predict(X)
tf.image.resize(
    var, [3,5], method=tf.image.ResizeMethod.BICUBIC, preserve_aspect_ratio=False,
    antialias=False, name=None
)
plt.imshow(var, cmap='gray')  # graph it
plt.show()  # display!


# # Loads the weights
# model1.load_weights(checkpoint_path)

# # Re-evaluate the model
# loss,acc = model1.evaluate(X,  y, verbose=2)
# print("Restored model, accuracy: {:5.2f}%".format(100*acc))
# print(model1.predict(X))
