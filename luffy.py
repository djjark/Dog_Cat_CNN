import cv2
import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt


DATADIR = "C:/Users/Diogo/Desktop/GitHub projects/Neural Networks/Dog_Cat_CNN/luffy"
IMG_SIZE = 150

training_data = []
def imagem():
    class_num = 0 # get the classification  (0 or a 1). 0=dog 1=cat
    for img in tqdm(os.listdir(DATADIR)):
        img_array = cv2.imread(os.path.join(DATADIR,img) ,cv2.IMREAD_GRAYSCALE)
        new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize to normalize data size
        training_data.append([new_array, class_num])  # add this to our training_data

imagem()

Xluffy = []
yluffy = []

for features,label in training_data:
    Xluffy.append(features)
    yluffy.append(label)

Xluffy = np.array(Xluffy).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

import pickle

pickle_out = open("Xluffy.pickle","wb")
pickle.dump(Xluffy, pickle_out)
pickle_out.close()

pickle_out = open("yluffy.pickle","wb")
pickle.dump(yluffy, pickle_out)
pickle_out.close()
# for s in range(5):
#     k=[]
#     for i in X[s]:
#         for j in range(IMG_SIZE):
#             k.append(i[j][0])
#     k = np.reshape(k,(IMG_SIZE,IMG_SIZE))
#     plt.imshow(k, cmap='gray')
#     plt.show()