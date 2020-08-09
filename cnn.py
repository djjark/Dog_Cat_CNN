import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import pickle
import math
import os
import cv2

IMG_SIZE = 150
pickle_in = open("Xluffy.pickle","rb")
Xluffy = pickle.load(pickle_in)

pickle_in = open("yluffy.pickle","rb")
yluffy = pickle.load(pickle_in)

Xluffy = Xluffy/255.0


pickle_in = open("X.pickle","rb")
X = pickle.load(pickle_in)

pickle_in = open("y.pickle","rb")
y = pickle.load(pickle_in)
X = X/255.0

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y = np.array(y)

Xluffy = np.array(Xluffy).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
yluffy= np.array(yluffy)



def create_model():
  model = tf.keras.models.Sequential([
    keras.layers.Conv2D(filters=64, kernel_size=(3, 3),activation='relu', input_shape=X.shape[1:]),
    keras.layers.MaxPool2D(pool_size=(2,2)),

    keras.layers.Conv2D(filters=64, kernel_size=(3, 3),activation='relu'),
    keras.layers.MaxPool2D(pool_size=(2,2)),

    keras.layers.Flatten(),
    keras.layers.Dense(1, activation='sigmoid')
  ])

  model.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy'])

  return model

def fit_model(model):
  # Train the model with the new callback
  model.fit(X, 
            y,  
            batch_size=32,
            epochs=10,
            validation_data=(X,y),
            callbacks=[cp_callback])  # Pass callback to training

checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)



def evalmodel(model1):
  loss,acc = model1.evaluate(X,  y, verbose=2)
  print("model accuracy: {:5.2f}%".format(100*acc))

def predicted(model1, Xluffy):
  predictions = model1.predict(Xluffy,batch_size=9)
  count=0
  for kq in predictions:
    if kq<0.5:
      print("| Imagem: "+str(count)+" | NN: Dog |"+" Real: "+("Dog | " if yluffy[count]==0 else "Cat | ")+" Percentagem: "+str(kq))
    else:
      print("| Imagem: "+str(count)+" | NN: Cat |"+" Real: "+ ("Dog | " if yluffy[count]==0 else "Cat | ")+" Percentagem: "+str(kq))
    k=[]
    for i in Xluffy[count]:
      for j in range(IMG_SIZE):
        k.append(i[j][0])
    k = np.reshape(k,(IMG_SIZE,IMG_SIZE))
    count+=1
    plt.imshow(k, cmap='gray')
    plt.show()

# create model
model1 = create_model()
# fit_model(model1)
# Loads the weights
model1.load_weights(checkpoint_path)
evalmodel(model1)
predicted(model1, Xluffy)
























































# print(predictions_single)
# plt.figure(figsize=(150,150))

# def plot_image(i, predictions_array, true_label, img):
#   predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
#   plt.grid(False)
#   plt.xticks([])
#   plt.yticks([])
#   k=[]
#   for a in Xluffy[i]:
#     for j in range(IMG_SIZE):
#       k.append(a[j][0])
#   k = np.reshape(k,(IMG_SIZE,IMG_SIZE))
#   plt.imshow(k, cmap='gray')

#   predicted_label = np.argmax(predictions_array)
#   if predicted_label == true_label:
#     color = 'blue'
#   else:
#     color = 'red'

#   plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
#                                 100*np.max(predictions_array),
#                                 class_names[true_label]),
#                                 color=color)

# def plot_value_array(i, predictions_array, true_label):
#   predictions_array, true_label = predictions_array[i], true_label[i]
#   plt.grid(False)
#   plt.xticks([])
#   plt.yticks([])
#   thisplot = plt.bar(range(len(class_names)), predictions_array, color="#777777")
#   plt.ylim([0, 1])
#   predicted_label = np.argmax(predictions_array)

#   thisplot[predicted_label].set_color('red')
#   thisplot[true_label].set_color('blue')

# plot_value_array(0, predictions_single, yluffy)
# _ = plt.xticks(range(len(class_names)), class_names, rotation=45)

# np.argmax(predictions_single[0].shape)

# # Plota o primeiro X test images, e as labels preditas, e as labels verdadeiras.
# # Colore as predições corretas de azul e as incorretas de vermelho.
# num_rows = 3
# num_cols = 3
# num_images = num_rows*num_cols
# plt.figure(figsize=(2*2*num_cols, 2*num_rows))
# for i in range(num_images):
#   plt.subplot(num_rows, 2*num_cols, 2*i+1)
#   plot_image(i, predictions, yluffy, Xluffy[i])
#   plt.subplot(num_rows, 2*num_cols, 2*i+2)
#   plot_value_array(i, predictions, yluffy)
# plt.show()


