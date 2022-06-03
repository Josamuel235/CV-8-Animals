import math
import numpy as np
import pandas as pd
import os
import shutil
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.5/bin")
from keras import regularizers
#from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.optimizers import RMSprop
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
#import cv2
from tensorflow.keras.callbacks import EarlyStopping
from keras.preprocessing import image



base= 'C:\MyJoseph\Certification\Data Analyst\Projects\ML_Practice\Data2\content\dataset'
train= os.path.join(base,'train')
test= os.path.join(base,'test')

train_ele= os.path.join(train ,'Elephant')
train_bear= os.path.join(train,'Bear')
#train_dog=os.path.join(train,'Dog')
train_gir=os.path.join(train,'Giraffe')
train_gor=os.path.join(train,'Gorilla')
train_kang=os.path.join(train,'Kangaroo')
train_pen=os.path.join(train,'Penguin')
train_rac=os.path.join(train,'Raccoon')
train_lion=os.path.join(train,'Lion')
train_zeb=os.path.join(train,'Zebra')

test_ele= os.path.join(test, 'Elephant')
test_bear= os.path.join(test,'Bear')
#test_dog=os.path.join(test,'Dog')
test_gir=os.path.join(test,'Giraffe')
test_gor=os.path.join(test,'Gorilla')
test_kang=os.path.join(test,'Kangaroo')
test_pen=os.path.join(test,'Penguin')
test_rac=os.path.join(test,'Raccoon')
test_lion=os.path.join(test,'Lion')
test_zeb=os.path.join(test,'Zebra')

print('total training elephants images:', len(os.listdir(train_ele)))
print('total training bears images:', len(os.listdir(train_bear)))
#print('total training dogs images:', len(os.listdir(train_dog)))
print('total training giraffe images:', len(os.listdir(train_gir)))
print('total training gorilla images:', len(os.listdir(train_gor)))
print('total training kangaroo images:', len(os.listdir(train_kang)))
print('total training penguin images:', len(os.listdir(train_pen)))
print('total training raccoon images:', len(os.listdir(train_rac)))
print('total training lion images:', len(os.listdir(train_lion)))
print('total training zebra images:', len(os.listdir(train_zeb)))
print('total test elephants images:', len(os.listdir(test_ele)))
print('total test bears images:', len(os.listdir(test_bear)))
#print('total test dogs images:', len(os.listdir(test_dog)))
print('total test giraffe images:', len(os.listdir(test_gir)))
print('total test gorilla images:', len(os.listdir(test_gor)))
print('total test kangaroo images:', len(os.listdir(test_kang)))
print('total test penguin images:', len(os.listdir(test_pen)))
print('total test raccoon images:', len(os.listdir(test_rac)))
print('total test lion images:', len(os.listdir(test_lion)))
print('total test zebra images:', len(os.listdir(test_zeb)))


train_data = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,)

test_data = ImageDataGenerator(rescale=1./255)

train_gen=train_data.flow_from_directory(
    train,
    target_size=(299,299),
    shuffle=True,
    class_mode='sparse'
    )

test_gen=test_data.flow_from_directory(
    test,
    target_size=(299,299),
    shuffle=True,
    class_mode='sparse'
    )

y_train=train_gen.classes
y_test=test_gen.classes


#### Model
incept = InceptionV3(input_shape=(299,299,3), weights='imagenet', include_top=False)
for layer in incept.layers:
    layer.trainable = False


x = layers.Conv2D(32, 3, activation='relu')(incept.output)
x = layers.MaxPooling2D(2,strides=(1,1), padding='same')(x)


x = layers.Conv2D(64, 3, activation='relu')(x)
x = layers.Conv2D(128, 3, activation='relu')(x)
x = layers.MaxPooling2D(2,strides=(1,1), padding='same')(x)

x= layers.Flatten()(x)

x= layers.Dense(128,kernel_regularizer=regularizers.l2(0.01) ,activation='relu')(x)
x = layers.Dropout(0.2)(x)

x= layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.4)(x)

x= layers.Dense(1024, activation='relu')(x)
x = layers.Dropout(0.5)(x)


prediction=layers.Dense(9,activation='softmax')(x)
model= Model(inputs=incept.input, outputs=prediction)
model.summary()

model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['sparse_categorical_accuracy'])

early_stop=EarlyStopping(monitor='val_loss',mode='min',verbose=1,patience=7)



history= model.fit(
      train_gen,
      #y_train,
      #batch_size=35,
      epochs=45,
      validation_data= (test_gen),#,y_test),
      callbacks=[early_stop],
      shuffle=True,
      verbose=2)

from tensorflow.keras.models import load_model

model.save("CVModel2.h5")
loaded_model = load_model("CVmodel2.h5")
print('FINISHED')
loss, accuracy = loaded_model.evaluate(test_gen)


