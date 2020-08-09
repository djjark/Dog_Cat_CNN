import cv2
import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt


DATADIR = "C:/Users/Diogo/Desktop/GitHub projects/Neural Networks/Dog_Cat_CNN"
CATEGORIES = ["luffy", "gtos"]
IMG_SIZE = 150
training_data = []
def imagem():
    for category in CATEGORIES:  # do dogs and cats

        path = os.path.join(DATADIR,category)  # create path to dogs and cats
        class_num = CATEGORIES.index(category)  # get the classification  (0 or a 1). 0=dog 1=cat
        count=0
        for img in tqdm(os.listdir(path)):  # iterate over each image per dogs and cats
            try:
                img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)  # convert to array
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize to normalize data size
                training_data.append([new_array, class_num])  # add this to our training_data
                count+=1
                if count>1000:
                    break
            except Exception as e:  # in the interest in keeping the output clean...
                pass
            #except OSError as e:
            #    print("OSErrroBad img most likely", e, os.path.join(path,img))
            #except Exception as e:
            #    print("general exception", e, os.path.join(path,img))

imagem()

import random

random.shuffle(training_data)

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