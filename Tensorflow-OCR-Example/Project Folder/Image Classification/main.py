import os
os.chdir('/home/username/PycharmProjects/pythonProject1')
os.environ["DISPLAY"] = ":0"
import PIL
from PIL import Image
import cv2
import numpy as np
import pandas as pd
from keras.layers import Dense, Flatten
from keras.layers.convolutional import Conv2D
from keras.models import Sequential
from tensorflow.keras.utils import to_categorical
import glob
import pyscreenshot as ImageGrab
import os
from keras.models import load_model
import string
import ranvar2m
from datetime import datetime
import time


#----- Load Current Model
model = load_model('/home/username/PycharmProjects/OCR_Example/my_model.h5')
#----- Input / Visual Area Data for Cropping to specified digit coordinates
x = 1029
y = 259
w = 1638
h = 412
im=ImageGrab.grab(backend='pil', bbox=(x,y,w,h))
img_np = np.array(im)
frame = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
cv2.imwrite('/home/username/project_folder/cap.png', frame)



#----- Crop digit images from project_folder capture
XDRimage = cv2.imread('/home/username/project_folder/cap.png')
crop_img = XDRimage[42:42+13, 941:941+13]
cv2.imwrite('/home/username/project_folder/var1/var1cap1.png', crop_img)
XDRimage = cv2.imread('/home/username/project_folder/cap.png')
crop_img = XDRimage[42:42+13, 960:960+14]
cv2.imwrite('/home/username/project_folder/var1/var1cap3.png', crop_img)
XDRimage = cv2.imread('/home/username/project_folder/cap.png')
crop_img = XDRimage[42:42+13, 973:973+14]
cv2.imwrite('/home/username/project_folder/var1/var1cap2.png', crop_img)
#- var2

XDRimage = cv2.imread('/home/username/project_folder/cap.png')
crop_img = XDRimage[131:131+11, 942:942+11]
cv2.imwrite('/home/username/project_folder/var2/var2cap2.png', crop_img)
XDRimage = cv2.imread('/home/username/project_folder/cap.png')
crop_img = XDRimage[131:131+11, 955:955+11]
cv2.imwrite('/home/username/project_folder/var2/var2cap1.png', crop_img)
XDRimage = cv2.imread('/home/username/project_folder/cap.png')
crop_img = XDRimage[131:131+11, 974:974+12]
cv2.imwrite('/home/username/project_folder/var2/var2cap3.png', crop_img)




#-var3
XDRimage = cv2.imread('/home/username/project_folder/cap.png')
crop_img = XDRimage[40:40+14, 471:471+14]
cv2.imwrite('/home/username/project_folder/var3/var3cap2.png', crop_img)
XDRimage = cv2.imread('/home/username/project_folder/cap.png')
crop_img = XDRimage[40:40+14, 490:490+14]
cv2.imwrite('/home/username/project_folder/var3/var3cap1.png', crop_img)
crop_img = XDRimage[42:42+11, 505:505+11]
cv2.imwrite('/home/username/project_folder/var3/var3cap3.png', crop_img)
#- var4
XDRimage = cv2.imread('/home/username/project_folder/cap.png')
crop_img = XDRimage[194:194+12, 1017:1017+13]
cv2.imwrite('/home/username/project_folder/var5/var43.png', crop_img)
XDRimage = cv2.imread('/home/username/project_folder/cap.png')


crop_img = XDRimage[195:195+11, 1051:1051+12]
cv2.imwrite('/home/username/project_folder/var5/var42.png', crop_img)
crop_img = XDRimage[194:194+12, 1031:1031+13]
cv2.imwrite('/home/username/project_folder/var5/var41.png', crop_img)



#- var5
XDRimage = cv2.imread('/home/username/project_folder/cap.png')
crop_img = XDRimage[386:386+10, 392:392+12]
cv2.imwrite('/home/username/project_folder/var5/var53.png', crop_img)
crop_img = XDRimage[386:386+10, 404:404+11]
cv2.imwrite('/home/username/project_folder/var5/var52.png', crop_img)
crop_img = XDRimage[386:386+10, 415:415+11]
cv2.imwrite('/home/username/project_folder/var5/var51.png', crop_img)


now = datetime.now()
current_time = now.strftime("%H:%M:%S")
chars1 = 'abcdef'
dig1 = '0123456789'
def id_generator(size=24, chars=chars1 + dig1):
    return ''.join(ranvar2m.choice(chars) for _ in range(size))
    
    
    
    
#---- Predict Digit Values
df_main_var1= pd.DataFrame()
#-- var1values
for img in glob.glob('/home/username/project_folder/var1/*.png'):
    print(img)
    new_img = []
    outname = id_generator(3)
    img_bin = cv2.imread(img, 0)
    resized_digit = cv2.resize(img_bin, (36, 36))
    padded_digit = np.pad(resized_digit, ((5, 5), (5, 5)), "constant", constant_values=0)
    for line in padded_digit:
        new_img.append(np.array(list(map(lambda x: 0 if x < 125 else 255, line))))
    new_img = np.array(list(map(lambda x: np.array(x), new_img)))
    prediction = model.predict(new_img.reshape(1, 46, 46, 1))
    var1pred = prediction
    df = pd.DataFrame(prediction)
    df_main_var1= df_main_var1.append(df)
dfvar1= df_main_var1.idxmax(axis=1)
#- var2 values
df_main_var2 = pd.DataFrame()
for img in glob.glob('/home/username/project_folder/var2/*.png'):
    print(img)
    new_img = []
    outname = id_generator(3)
    img_bin = cv2.imread(img, 0)
    resized_digit = cv2.resize(img_bin, (36, 36))
    padded_digit = np.pad(resized_digit, ((5, 5), (5, 5)), "constant", constant_values=0)
    for line in padded_digit:
        new_img.append(np.array(list(map(lambda x: 0 if x < 125 else 255, line))))
    new_img = np.array(list(map(lambda x: np.array(x), new_img)))
    var2traindigit = new_img.reshape(1, 46, 46, 1)
    prediction = model.predict(var2traindigit.reshape(1, 46, 46, 1))
  #  var1pred = prediction
    df = pd.DataFrame(prediction)
    df_main_var2 = df_main_var2.append(df)
dfvar2 = df_main_var2.idxmax(axis=1)
- var5 values
df_main_var5 = pd.DataFrame()
for img in glob.glob('/home/username/project_folder/var5/*.png'):
    print(img)
    img_bin = cv2.imread(img, 0)
    thresh = cv2.threshold(img_bin, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    resized_digit = cv2.resize(thresh, (36, 36))
    padded_digit = np.pad(resized_digit, ((5, 5), (5, 5)), "constant", constant_values=0)
    prediction = model.predict(padded_digit.reshape(1, 46, 46, 1))
    df = pd.DataFrame(prediction)
    df_main_var5 = df_main_var5.append(df)
dfvar5 = df_main_var5.idxmax(axis=1)
#- var3 values
df_main_var3 = pd.DataFrame()
for img in glob.glob('/home/username/project_folder/var3/*.png'):
    print(img)
    new_img = []
    outname = id_generator(3)
    img_bin = cv2.imread(img, 0)
    resized_digit = cv2.resize(img_bin, (36, 36))
    padded_digit = np.pad(resized_digit, ((5, 5), (5, 5)), "constant", constant_values=0)
    for line in padded_digit:
        new_img.append(np.array(list(map(lambda x: 0 if x < 125 else 255, line))))
    new_img = np.array(list(map(lambda x: np.array(x), new_img)))
    prediction = model.predict(new_img.reshape(1, 46, 46, 1))

    df = pd.DataFrame(prediction)
    df_main_var3 = df_main_var3.append(df)
dfvar3 = df_main_var3.idxmax(axis=1)

df_main_var5 = pd.DataFrame()
predarr = []
for img in glob.glob('/home/username/project_folder/var5/*.png'):
    print(img)
    new_img = []
    outname = id_generator(3)
    img_bin = cv2.imread(img, 0)
    resized_digit = cv2.resize(img_bin, (36, 36))
    padded_digit = np.pad(resized_digit, ((5, 5), (5, 5)), "constant", constant_values=0)
    for line in padded_digit:
        new_img.append(np.array(list(map(lambda x: 0 if x < 125 else 255, line))))
    new_img = np.array(list(map(lambda x: np.array(x), new_img)))
    #cv2.imwrite('/home/username/project_folder/var5/digit_checker/thresh_{}.png'.format(outname), new_img)
    prediction = model.predict(new_img.reshape(1, 46, 46, 1))
    predarr.append(prediction)
    df = pd.DataFrame(prediction)
    df_main_var5 = df_main_var5.append(df)
dfvar5 = df_main_var5.idxmax(axis=1)

#-- Write to project_folder Predicted Digits to .csv
dfvar1.to_csv('/home/username/project_folder/var1/var1.csv')
dfvar2.to_csv('/home/username/project_folder/var2/var2.csv')
dfvar4.to_csv('/home/username/project_folder/var5/var5.csv')
dfvar3.to_csv('/home/username/project_folder/var3/var3.csv')
dfvar5.to_csv('/home/username/project_folder/var5/var5.csv')


print('RECOGNIZED DIGITS FOR ALERTS:')
print('')
#print('')
print('var5')
print(dfvar5)
print('var3')
print(dfvar3)
print('dfvar2')
print(dfvar2)
print('var1')
print(dfvar1)
print('var4')
print(dfvar4)



