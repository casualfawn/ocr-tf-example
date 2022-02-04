from tqdm import tqdm
import os
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
from keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.layers import Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
import tensorflow as tf
import random as rn


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

datagen = ImageDataGenerator(
        rotation_range=5,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.25,
        zoom_range=0.15,
        #horizontal_flip=True,
        fill_mode='constant')

img = load_img('/home/username/Project Folder/digits/0s_dir/0s.png')  # this is a PIL image
x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)
i = 0
for batch in datagen.flow(x, batch_size=1,
                          save_to_dir='/home/username/Project Folder/digits/0s_dir/', save_prefix='train0s', save_format='png'):
    i += 1
    if i > 600:
        break  


img = load_img('/home/username/Project Folder/digits/1s_dir/1s.png')  # this is a PIL image
x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)
i = 0
for batch in datagen.flow(x, batch_size=1,
                          save_to_dir='/home/username/Project Folder/digits/1s_dir/', save_prefix='train1s', save_format='png'):
    i += 1
    if i > 600:
        break  

img = load_img('/home/username/Project Folder/digits/2s_dir/2s.png')  # this is a PIL image
x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)
i = 0
for batch in datagen.flow(x, batch_size=1,
                          save_to_dir='/home/username/Project Folder/digits/2s_dir/', save_prefix='train2s', save_format='png'):
    i += 1
    if i > 600:
        break  

img = load_img('/home/username/Project Folder/digits/3s_dir/3s.png')  # this is a PIL image
x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)
i = 0
for batch in datagen.flow(x, batch_size=1,
                          save_to_dir='/home/username/Project Folder/digits/3s_dir/', save_prefix='train3s', save_format='png'):
    i += 1
    if i > 600:
        break 

img = load_img('/home/username/Project Folder/digits/4s_dir/4s.png')  # this is a PIL image
x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)
i = 0
for batch in datagen.flow(x, batch_size=1,
                          save_to_dir='/home/username/Project Folder/digits/4s_dir/', save_prefix='train4s', save_format='png'):
    i += 1
    if i > 600:
        break  
        
img = load_img('/home/username/Project Folder/digits/5s_dir/5s.png')  # this is a PIL image
x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)
i = 0
for batch in datagen.flow(x, batch_size=1,
                          save_to_dir='/home/username/Project Folder/digits/5s_dir/', save_prefix='train5s', save_format='png'):
    i += 1
    if i > 600:
        break  

img = load_img('/home/username/Project Folder/digits/6s_dir/6s.png')  # this is a PIL image
x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)
i = 0
for batch in datagen.flow(x, batch_size=1,
                          save_to_dir='/home/username/Project Folder/digits/6s_dir/', save_prefix='train6s', save_format='png'):
    i += 1
    if i > 600:
        break 

img = load_img('/home/username/Project Folder/digits/7s_dir/7s.png')  # this is a PIL image
x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)
i = 0
for batch in datagen.flow(x, batch_size=1,
                          save_to_dir='/home/username/Project Folder/digits/7s_dir/', save_prefix='train7s', save_format='png'):
    i += 1
    if i > 600:
        break 

img = load_img('/home/username/Project Folder/digits/8s_dir/8s.png')  # this is a PIL image
x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)
i = 0
for batch in datagen.flow(x, batch_size=1,
                          save_to_dir='/home/username/Project Folder/digits/8s_dir/', save_prefix='train8s', save_format='png'):
    i += 1
    if i > 600:
        break

img = load_img('/home/username/Project Folder/digits/9s_dir/9s.png')  # this is a PIL image
x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)
i = 0
for batch in datagen.flow(x, batch_size=1,
                          save_to_dir='/home/username/Project Folder/digits/9s_dir/', save_prefix='train9s', save_format='png'):
    i += 1
    if i > 600:
        break  

