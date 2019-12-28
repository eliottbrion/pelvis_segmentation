# This file is inspired from https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import numpy as np
import keras
from utils import *
import random
import pickle
from scipy import ndimage

image_size = [160, 160, 128]
spacing = [1.2, 1.2, 1.5]
organs_names = ['bladder', 'rectum', 'prostate']

class DataGeneratorTrain(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, normalization_params, params):
        'Initialization'
        self.dim = tuple(image_size)
        self.batch_size = params['batch_size']
        self.list_IDs = list_IDs
        self.n_channels = 1
        self.shuffle = True
        self.normalization_params = normalization_params
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, Y = self.__data_generation(list_IDs_temp)
        return X, Y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        n_organs = len(organs_names)
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        Y = np.empty((self.batch_size,*self.dim,n_organs+1))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            shears = np.array([0.02*random.uniform(-1,1) for _ in range(6)])
            angles = np.array([5*random.uniform(-1,1) for _ in range(3)])
            shifts = np.array([0.05*random.uniform(-1,1)*image_size[i] for i in range(3)])

            im = np.load('data/' + ID + '-image.npy')
            im = (im-self.normalization_params['mean'])/self.normalization_params['std']
            im = image_transform(im, shears, angles, shifts, order=3)
            X[i,] = np.expand_dims(im, axis=-1)

            # Store class
            masks = np.load('data/' + ID + '-mask.npy')
            masks_trans = np.zeros((*self.dim,n_organs+1))
            for organ_num in range(n_organs):
                masks_trans[:,:,:,organ_num] = image_transform(masks[:,:,:,organ_num], shears, angles, shifts, order=0)
            masks_trans[:,:,:,-1] = 1 
            for organ_num in range(n_organs):
                masks_trans[:,:,:,-1] = masks_trans[:,:,:,-1] - masks_trans[:,:,:,organ_num]
            Y[i,] = masks_trans

        return X, Y
    
class DataGeneratorVal(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, normalization_params, params):
        'Initialization'
        self.dim = tuple(image_size)
        self.batch_size = params['batch_size']
        self.list_IDs = list_IDs
        self.n_channels = 1
        self.shuffle = False
        self.normalization_params = normalization_params
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, Y = self.__data_generation(list_IDs_temp)

        return X, Y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        n_organs = len(organs_names)
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        Y = np.empty((self.batch_size,*self.dim,n_organs+1))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            im = np.load('data/' + ID + '-image.npy')
            im = (im-self.normalization_params['mean'])/self.normalization_params['std']
            X[i,] = np.expand_dims(im, axis=-1)

            # Store class
            Y[i,] = np.load('data/' + ID + '-mask.npy')

        return X, Y
    
def image_transform(image, shears, angles, shifts, order):
    shear_matrix = np.array([[1, shears[0], shears[1], 0],
                         [shears[2], 1, shears[3], 0],
                         [shears[4], shears[5], 1, 0],
                         [0, 0, 0, 1]])

    shift_matrix = np.array([[1, 0, 0, shifts[0]],
                         [0, 1, 0, shifts[1]],
                         [0, 0, 1, shifts[2]],
                         [0, 0, 0, 1]])

    offset = np.array([[1, 0, 0, int(image_size[0]/2)],
                   [0, 1, 0, int(image_size[1]/2)],
                   [0, 0, 1, int(image_size[2]/2)],
                   [0, 0, 0, 1]])

    offset_opp = np.array([[1, 0, 0, -int(image_size[0]/2)],
                   [0, 1, 0, -int(image_size[1]/2)],
                   [0, 0, 1, -int(image_size[2]/2)],
                   [0, 0, 0, 1]])

    angles = np.deg2rad(angles)
    rotx = np.array([[1, 0, 0, 0],
                 [0, np.cos(angles[0]), -np.sin(angles[0]), 0],
                 [0, np.sin(angles[0]), np.cos(angles[0]), 0],
                 [0, 0, 0, 1]])
    roty = np.array([[np.cos(angles[1]), 0, np.sin(angles[1]), 0],
                 [0, 1, 0, 0],
                 [-np.sin(angles[1]), 0, np.cos(angles[1]), 0],
                 [0, 0, 0, 1]])
    rotz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0, 0],
                 [np.sin(angles[2]), np.cos(angles[2]), 0, 0],
                 [0, 0, 1, 0],
                 [0, 0, 0, 1]])
    rotation_matrix = offset_opp.dot(rotz).dot(roty).dot(rotx).dot(offset)
    affine_matrix = shift_matrix.dot(rotation_matrix).dot(shear_matrix)
    return ndimage.interpolation.affine_transform(image, affine_matrix, order=order, mode='nearest')
    
