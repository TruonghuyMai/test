# Common
import os
import keras
import datetime
import numpy as np
from tqdm import tqdm 
from glob import glob
import tensorflow as tf
import tensorflow.image as tfi

# Data
from keras.preprocessing.image import load_img, img_to_array 

# Data Viz
import matplotlib.pyplot as plt

# Model
from keras.layers import ReLU
from keras.layers import Layer
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import Dropout
from keras.layers import MaxPool2D
from keras.layers import LeakyReLU
from keras.layers import concatenate
from keras.layers import ZeroPadding2D
from keras.layers import Conv2DTranspose
from keras.initializers import RandomNormal
from keras.layers import BatchNormalization
from tensorflow.keras.optimizers import Adam
from keras.models import Sequential, Model, load_model

# Model Viz
from tensorflow.keras.utils import plot_model

# Model Losses
from keras.losses import BinaryCrossentropy
from keras.losses import MeanAbsoluteError

#trainA = RGB image
#trainB = NIR image
#images = color = NIR
#mask = gray = RGB 
def load_data(trainA_path,trainB_path, trim=1000):
    #color_paths = sorted(glob(trainB_path + "*.png"))[:trim]
    #gray_paths =  sorted(glob(trainA_path + "*.png"))[:trim]
    #color_paths = sorted(glob(trainB_path + "*.png"))[:2]
    #gray_paths =  sorted(glob(trainA_path + "*.png"))[:2]
    color_paths=[]
    color_paths.append(trainA_path)
    gray_paths=color_paths
    #print(color_paths)
    #for path in color_paths:
        #gray_paths.append(path.replace('train_B/', 'train/'))
    
    images = np.zeros(shape=(len(color_paths), 256, 256, 3))
    masks = np.zeros(shape=(len(gray_paths), 256, 256, 3))
    

    i=0
    for color_path, gray_path in tqdm(zip(color_paths, gray_paths), desc="Data"):
        
        image = tf.cast(img_to_array(load_img(color_path)), tf.float32)
        mask = tf.cast(img_to_array(load_img(gray_path)), tf.float32)
        
        images[i] = (tfi.resize(image,(256,256)))/255.
        masks[i] = (tfi.resize(mask,(256,256)))/255.

        i+=1
    
    return images, masks
  
def show_image(image, title=None, alpha=1.0):
    plt.imshow(image, alpha=alpha)
    if title is not None:
        plt.title(title)
    plt.axis('off')
def show_image(image, title=None, alpha=1.0):
    plt.imshow(image, alpha=alpha)
    if title is not None:
        plt.title(title)
    plt.axis('off')
 def upsample(filters, apply_dropout=False):

    model = Sequential()
    model.add(Conv2DTranspose(
        filters,
        kernel_size=4,
        strides=2,
        padding='same',
        kernel_initializer='he_normal',
        use_bias=False))
    model.add(BatchNormalization())
    
    if apply_dropout:
        model.add(Dropout(0.5))
    
    model.add(ReLU())
    
def Generator():

    inputs = Input(shape=(256,256,3), name="InputLayer")

    down_stack = [
        downsample(64, apply_batchnorm=False),
        downsample(128),
        downsample(256),
        downsample(512),
        downsample(512),
        downsample(512),
        downsample(512),
    ]
    encoding = downsample(512)

    up_stack = [
        upsample(512, apply_dropout=True),
        upsample(512, apply_dropout=True),
        upsample(512, apply_dropout=True),
        upsample(512),
        upsample(256),
        upsample(128),
        upsample(64),
    ]

    x = inputs 
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)
    
    x = encoding(x)

    skips = reversed(skips)
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = concatenate([x, skip])
    
    init = RandomNormal(stddev=0.02)
    out = Conv2DTranspose(3, kernel_size=4, strides=2, kernel_initializer=init, activation='tanh', padding='same')

    out = out(x)

    gen = Model(
        inputs=inputs,
        outputs=out,
        name="Generator"
    )
    return gen
    return model
    
pathA = '/kaggle/input/deepnir-nir-rgb-capsicum/capsicums_pix2pixHD/test_A/jai_04-12-14_T1614_pic000010-0000.png'
pathB = '/kaggle/input/deepnir-nir-rgb-capsicum/capsicums_pix2pixHD/test_B/'
color_images, gray_images = load_data(pathA,pathB)
