import os

import numpy as np
import keras 

from keras.utils import to_categorical, Sequence
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

class ImageSequence(Sequence): 
    '''Extends the keras Sequence class. It takes in a series of indicies and it pulls the pictures from the folder specified
    it generates batches with the inputed batch size that have been preprocessed by the inputed preprocess function
    '''
    def __init__(self,labeldict, batch_size, train_indicies,preprocessing_func,folder = 'data/initial_images/'):
        self.NUM_CLASSES = 228
        self.image_folder = folder
        self.indicies = train_indicies
        self.labels = labeldict
        self.batch_size = batch_size
        self.length = len(labeldict)
        #self.data_generator = data_generator
        self.preprocessing_func = preprocessing_func
    def __len__(self):
        return (len(self.indicies)//self.batch_size)
    
    def __getitem__(self, idx):
        '''gets and processes one batch
        '''
        inds = self.indicies[idx * self.batch_size : idx * (self.batch_size+1)]
        X = np.array([self.get_img_array(n) for n in inds])
        y = np.array([self.get_label(n) for n in inds])   
        return X, y
    
    def get_label(self,n):
        '''helper function to get one label'''
        y = [int(x) - 1 for x in self.labels[n]['labelId']]
        y = np.array(keras.utils.to_categorical(y, num_classes = self.NUM_CLASSES).sum(axis = 0))
        return y
        
    def get_img_array(self,n):
        '''helper function to get one image and process it'''
        img = image.load_img('{}{}.jpg'.format(self.image_folder,n),grayscale=False,target_size=(224,224,3))
        image.img_to_array(img)
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis = 0)
        x = x.reshape((224,224,3))
        x = self.preprocessing_func(x)
        return x

    
def create_train_val_inds(val_size,folder = 'data/initial_images/'):
        '''helper function to make a train/val split
        '''
        inds = [x[:-4] for x in os.listdir(folder)]
        val_inds = np.random.choice(inds,size = val_size)
        train_inds = np.array([x for x in inds if x not in val_inds])
        return train_inds, val_inds
    
def create_validation(labels, validation_inds, preprocess_func, folder = 'data/initial_images/',NUM_CLASSES = 228):  
    '''creates and processes the validation set 
    '''
    X,y = [],[]
    for n in validation_inds:   
        img = image.load_img('{}{}.jpg'.format(folder,n),grayscale=False,target_size=(224,224,3))
        image.img_to_array(img)
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis = 0)
        x = preprocess_func(x)
        X.append(x.reshape((224,224,3)))
        label = labels[n]['labelId']
        label = np.array([int(l) - 1 for l in label])
        label = keras.utils.to_categorical(label, num_classes = NUM_CLASSES).sum(axis = 0)
        y.append(label)
    X,y = np.array(X), np.array(y)
    return X,y

def create_sequence_and_val(labels, val_size,batch_size,preprocess_func,folder = 'data/initial_images/'):
    '''creates a ImageSequence object as well as a validation set for the keras 
    fit_generator method
    '''
    
    train_inds,val_inds = create_train_val_inds(val_size, folder = folder)
    
    Xval,yval = create_validation(labels,val_inds,preprocess_func,folder = folder)
    
    train_sequence = ImageSequence(labels, batch_size, train_inds, preprocess_func, folder = folder)
    
    return train_sequence, Xval, yval
