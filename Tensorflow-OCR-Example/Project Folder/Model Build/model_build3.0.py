from PIL import Image
import cv2
from keras.layers import Dense, Flatten
from keras.layers.convolutional import Conv2D
from keras.models import Sequential
from tensorflow.keras.utils import to_categorical
import glob
import pyscreenshot as ImageGrab
import os
from tqdm import tqdm
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score,precision_score,recall_score,confusion_matrix,roc_curve,roc_auc_score
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.layers import Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
import tensorflow as tf
import random as rn
import string





#-- Set images of digits directories
train0_DIR = '/home/username/Project Folder/digits/0s_dir'
train1_DIR = '/home/username/Project Folder/digits/1s_dir'
train2_DIR = '/home/username/Project Folder/digits/2s_dir'
train3_DIR = '/home/username/Project Folder/digits/3s_dir'
train4_DIR = '/home/username/Project Folder/digits/4s_dir'
train5_DIR = '/home/username/Project Folder/digits/5s_dir'
train6_DIR = '/home/username/Project Folder/digits/6s_dir'
train7_DIR = '/home/username/Project Folder/digits/7s_dir'
train8_DIR = '/home/username/Project Folder/digits/8s_dir'
train9_DIR = '/home/username/Project Folder/digits/9s_dir'

#-- Function for labels
def assign_label(img, digit):
	return digit
def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))
#-- Function for creating train/test datasets
def maketrain(digit, DIR):
    for img in tqdm(os.listdir(DIR)):
        path = os.path.join(DIR, img)
        label = assign_label(img, digit)
        outname = id_generator(3)
        new_img = []
        img_bin = cv2.imread(path, 0)
        resized_digit = cv2.resize(img_bin, (36, 36))
        padded_digit = np.pad(resized_digit, ((5, 5), (5, 5)), "constant", constant_values=0)
        for line in padded_digit:
            new_img.append(np.array(list(map(lambda x: 0 if x < 188 else 255, line))))
        new_img = np.array(list(map(lambda x: np.array(x), new_img)))
     
        new_img = new_img.reshape(1, 46, 46)
        x.append(np.array(new_img))
        y.append(str(label))



#-- compile train/test datasets and format
x = []
y = []
maketrain('train0s', train0_DIR)
maketrain('train1s', train1_DIR)
maketrain('train2s', train2_DIR)
maketrain('train3s', train3_DIR)
maketrain('train4s', train4_DIR)
maketrain('train5s', train5_DIR)
maketrain('train6s', train6_DIR)
maketrain('train7s', train7_DIR)
maketrain('train8s', train8_DIR)
maketrain('train9s', train9_DIR)
le = LabelEncoder()
y = le.fit_transform(y)
x = np.array(x)
#x = x/255
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size = 0.25)
X_test = x_test.reshape(len(x_test), 46, 46, 1)
X_train = x_train.reshape(len(x_train), 46, 46, 1)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


#--build model
model = Sequential()

## Declare the layers
layer_1 = Conv2D(32, kernel_size=3, activation="relu", input_shape=(46, 46, 1))
layer_2 = Conv2D(64, kernel_size=3, activation="relu")
layer_3 = Flatten()
layer_4 = Dense(10, activation="softmax")

## Add layers
model.add(layer_1)
model.add(layer_2)
model.add(layer_3)
model.add(layer_4)

#- Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#- Train model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20000)

#- Save model
model.save('my_model.h5')
#model = model
