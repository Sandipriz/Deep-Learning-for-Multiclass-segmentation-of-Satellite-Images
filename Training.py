#*********************TRAINING CODE*******************************




import PIL
from PIL import Image
import matplotlib.pyplot as plt
from libtiff import TIFF
from libtiff import TIFFfile, TIFFimage
from scipy.misc import imresize
import numpy as np
import math
import glob
import cv2
import os
import skimage.io as io
import skimage.transform as trans
import keras
from keras import applications
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as keras
import re




#To convert tensorflow metrics into keras metrics
def as_keras_metric(method):
    import functools
    from keras import backend as K
    import tensorflow as tf
    @functools.wraps(method)
    def wrapper(self, args, **kwargs):
        """ Wrapper for turning tensorflow metrics into keras metrics """
        value, update_op = method(self, args, **kwargs)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([update_op]):
            value = tf.identity(value)
        return value
    return wrapper

#Metric used to evaluate model
def iou(y_true, y_pred, epsilon = 1e-8):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    union = K.sum(y_true,-1) + K.sum(y_pred,-1) - intersection
    iou_acc = (intersection + epsilon) / (union + epsilon)
    return iou_acc


#Mean IoU
def tf_mean_iou(y_true, y_pred, num_classes=8):
    return tf.metrics.mean_iou(y_true, y_pred, num_classes)

mean_iou = as_keras_metric(tf_mean_iou)


#To numerically arrange images in the dataset
def numericalSort(value):
    numbers = re.compile(r'(\d+)')
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

#*********************************************************************************************************************************************
#PATH TO IMAGES IN THE DATASET
# List of file names of actual Satellite images for traininig 
filelist_trainx = sorted(glob.glob('PATH-TO-FOLDER-CONTAINING-ORIGINAL-TRAINING-IMAGES/*.tif'), key=numericalSort)
# List of file names of classified images for traininig 
filelist_trainy = sorted(glob.glob('PATH-TO-FOLDER-CONTAINING-GROUND-TRUTHS-OF-TRAINING-IMAGES/*.tif'), key=numericalSort)

# List of file names of actual Satellite images for validation
filelist_testx = sorted(glob.glob('PATH-TO-FOLDER-CONTAINING-ORIGINAL-VALIDATION-IMAGES/*.tif'), key=numericalSort)
# List of file names of classified images for validation                                        
filelist_testy = sorted(glob.glob('PATH-TO-FOLDER-CONTAINING-GROUND-TRUTHS-OF-VALIDATION-IMAGES/*.tif'), key=numericalSort)

#*********************************************************************************************************************************************

# Padding at bottom and left to crop img into 128*128 for training
def padding(img, w, h, c, crop_size, stride, n_h, n_w):
    
    w_extra = w - ((n_w-1)*stride)
    w_toadd = crop_size - w_extra
    
    h_extra = h - ((n_h-1)*stride)
    h_toadd = crop_size - h_extra
    
    img_pad = np.zeros(((h+h_toadd), (w+w_toadd), c))
    img_pad = np.pad(img, [(0, h_toadd), (0, w_toadd), (0,0)], mode='constant')
    
    return img_pad
    

# Adding pixels to make the image with shape in multiples of stride
def add_pixals(img, h, w, c, n_h, n_w, crop_size, stride):
        
    w_extra = w - ((n_w-1)*stride)
    w_toadd = crop_size - w_extra

    h_extra = h - ((n_h-1)*stride)
    h_toadd = crop_size - h_extra

    img_add = np.zeros(((h+h_toadd), (w+w_toadd), c))
        
    img_add[:h, :w,:] = img
    img_add[h:, :w,:] = img[:h_toadd,:, :]
    img_add[:h,w:,:] = img[:,:w_toadd,:]
    img_add[h:,w:,:] = img[h-h_toadd:h,w-w_toadd:w,:]
            
    return img_add  


#Function to crop images into a size of 128*128
def crops(a, crop_size = 128):
    
    stride = 32

    croped_images = []
    h, w, c = a.shape
    
    n_h = int(int(h/stride))
    n_w = int(int(w/stride))
    
    # Padding using the padding function 
    #a = padding(a, w, h, c, crop_size, stride, n_h, n_w)
    a = add_pixals(a, h, w, c, n_h, n_w, crop_size, stride)
    
    
    # Slicing the image into 128*128 crops with a stride of 32
    for i in range(n_h-1):
        for j in range(n_w-1):
            crop_x = a[(i*stride):((i*stride)+crop_size), (j*stride):((j*stride)+crop_size), :]
            croped_images.append(crop_x)
    return croped_images

#Function to make an array of cropped images
def make_array(filelist,length):
    new_list = []
    for fname in filelist[:length]:
        # Reading the image
        tif = TIFF.open(fname)
        image = tif.read_image()

        # Padding as required and cropping
        crops_list = crops(image)
    	  #print(len(crops_list))
        new_list = new_list + crops_list
    
	    	# Array of all the cropped Images    
    final_list = np.asarray(new_list)
    return final_list, new_list

trainx, trainx_list = make_array(filelist_trainx,13)
trainy, trainy_list = make_array(filelist_trainy,13)
testx, testx_list = make_array(filelist_testx,1)
testy, testy_list = make_array(filelist_testy,1)

#***************************************  MODEL USED ***************************************************

#Block at each horizontal level
def block(input_tensor, filters):
    
    x =  Conv2D(filters, 3, activation = 'relu', padding = 'same', kernel_initializer = 'random_normal')(input_tensor)
    x =  Conv2D(filters, 3, activation = 'relu', padding = 'same', kernel_initializer = 'random_normal')(x) 
    x =  BatchNormalization()(x)
    x1 = SeparableConv2D(filters, 3, activation = 'relu', padding = 'same', depthwise_initializer = 'random_normal',pointwise_initializer='random_normal')(x)
    x1 = SeparableConv2D(filters, 3, activation = 'relu', padding = 'same', depthwise_initializer = 'random_normal',pointwise_initializer='random_normal')(x1)
    x1 = BatchNormalization()(x1)
    x =  Add()([x,x1])
    return x



def model_mod(shape = (None,None,4)):
    
    # Left side of the U-Net
    inputs = Input(shape)
    conv1 = block(inputs,64)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = block(pool1,128)
    drop2 = Dropout(0.2)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(drop2)
    conv3 = block(pool2,256)
    drop3 = Dropout(0.2)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(drop3)
    conv4 = block(pool3,512)
    drop4 = Dropout(0.25)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    # Bottom of the U-Net
    conv5 = block(pool4,1024)
    drop5 = Dropout(0.4)(conv5)
 
    # Upsampling Starts, right side of the U-Net
    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same')(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = block(merge6,512)
    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = block(merge7,256)
    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = block(merge8, 128)
    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = block(merge9,64)
    conv9 = Conv2D(16, 3, activation = 'relu', padding = 'same')(conv9)
    conv9 = BatchNormalization()(conv9)
    # Output layer of the U-Net with a softmax activation
    conv10 = Conv2D(9, 1, activation = 'softmax')(conv9)

    model = Model(input = inputs, output = conv10)
    model.compile(optimizer = Nadam(lr = 0.001), loss = 'categorical_crossentropy', metrics = ['accuracy', iou])
    model.summary()
    return model

model = model_mod()


#****************************************************************************************************************

#Color code for each class
color_dict = {0: (0, 0, 0),
              1: (0, 125, 0),
              2: (150, 80, 0),
              3: (255, 255, 0),
              4: (100, 100, 100),
              5: (0, 255, 0),
              6: (0, 0, 150),
              7: (150, 150, 255),
              8: (255, 255, 255)}

#Converting images from RGB to One-hot
def rgb_to_onehot(rgb_arr, color_dict):
    num_classes = len(color_dict)
    shape = rgb_arr.shape[:2]+(num_classes,)
    #print(shape)
    arr = np.zeros( shape, dtype=np.int8 )
    for i, cls in enumerate(color_dict):
        arr[:,:,i] = np.all(rgb_arr.reshape( (-1,3) ) == color_dict[i], axis=1).reshape(shape[:2])
    return arr

#Converting images from One-hot to RGB 
def onehot_to_rgb(onehot, color_dict):
    single_layer = np.argmax(onehot, axis=-1)
    output = np.zeros( onehot.shape[:2]+(3,) )
    for k in color_dict.keys():
        output[single_layer==k] = color_dict[k]
    return np.uint8(output)

#Function to convert trainy and testy to one hot encoded format
def conv_to_onehot():
    trainy_hot = []
    for i in range(trainy.shape[0]):
        hot_img = rgb_to_onehot(trainy[i], color_dict)
        trainy_hot.append(hot_img)
    trainy_hot = np.asarray(trainy_hot)

	
    testy_hot = []
    for i in range(testy.shape[0]):
        hot_img = rgb_to_onehot(testy[i], color_dict)
        testy_hot.append(hot_img)
    testy_hot = np.asarray(testy_hot)

    return trainy_hot, testy_hot


trainy_hot,testy_hot = conv_to_onehot()





#*****************TRAINING THE MODEL / FITTING******************************

history = model.fit(trainx, trainy_hot, epochs=25, validation_data = (testx, testy_hot),batch_size=4, verbose=2)
model.save("/content/model_onehot.h5") #Path to save weights file

#****************************************************************************

#GRAPHS FOR ACCURACY, LOSS AND IoU
# list all data in history

print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig('acc_plot.png')
plt.show()
plt.close()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'val'], loc='upper right')
plt.savefig('loss_plot.png')
plt.show()
plt.close()


plt.plot(history.history['iou'])
plt.plot(history.history['val_iou'])
plt.title('Model iou')
plt.ylabel('iou')
plt.xlabel('Epoch')
plt.legend(['train', 'val'], loc='upper right')
plt.savefig('iou_plot.png')
plt.show()
plt.close()



#**********************END OF CODE****************************************









