from keras.models import Model, Sequential
from keras.layers import Input, concatenate, Conv3D, MaxPooling3D, Conv3DTranspose, Flatten, Dense
import tensorflow as tf
import numpy as np
import os
import pickle
from scipy import ndimage
from data_generators import *

from models import *
from utils import *

image_size = [160,160,128]
spacing = [1.2, 1.2, 1.5]
organs_names = ['bladder', 'rectum', 'prostate']

def train(partition, previous_dir, gpu, dest_dir, params):
    # partition: dictionary with keys training, validation and test. partition['training'] is a list of string with the path of the training images. Similarly for validation and test
    # previous_dir: path to the directory of a previously trained model. The directory contains weight.h5 with the model's weights, the directory params and the directory history. If previous_dir==None, starts training from scratch.
    # gpu: id of gpu to be used (integer)
    # dest_dir: destination of the directory where to put the results
    # params: dictionary with the training's parameters
    
    # Organs
    n_organs = len(organs_names)
    
    # Set gpu and seeds
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    from numpy.random import seed
    seed(1)
    from tensorflow import set_random_seed
    set_random_seed(1)
    
    # Create folder and subfolder
    if not os.path.exists(dest_dir + '/' + params2name(params)):
        os.makedirs(dest_dir + '/' + params2name(params))
        pickle.dump( params, open( dest_dir + '/' + params2name(params) + '/params.p', "wb" ) )
        
    crossvalidation_dir = dest_dir + '/' + params2name(params) + '/firstval_' + partition['val'][0]
    if not os.path.exists(crossvalidation_dir):
        os.makedirs(crossvalidation_dir)
        pickle.dump( params, open( crossvalidation_dir + '/params.p', "wb" ) )
        pickle.dump( partition, open( crossvalidation_dir + '/partition.p', "wb" ) )
        np.save(crossvalidation_dir + '/previous_dir.npy', previous_dir)
    
    # Create or load model
    if previous_dir==None:
        print('Starting training from scratch.')
        params_previous = {'epochs': 0}
        model = unet_3d(params)
        hist1 = None # no history from previous model
    else:
        params_previous = pickle.load( open( previous_dir + '/params.p', "rb" ) )
        print('Resuming training of a model trained for ' + str(params_previous['epochs']) + ' epochs.')
        hist1 = pickle.load( open( previous_dir + '/history.p', "rb" ) ) # history from previous model
        if params['loss']=='gen_dice_loss':
            co = {'gen_dice_loss': gen_dice_loss, 'b': b, 'r': r}
        elif params['loss']=='categorical_crossentropy':
            co = {'b': b, 'r': r, 'p': p}
        elif params['loss']=='dl':
            co = {'dl':dl, 'b': b, 'r': r, 'p': p}
        model = load_model(previous_dir + '/weights.h5', custom_objects=co)
        
    # Compute normalization parameters
    # Adapted from https://en.wikipedia.org/wiki/Standard_deviation#Rapid_calculation_methods
    s1 = 0
    s2 = 0
    N = len(partition['train'])*np.prod(image_size)
    for filepath in partition['train']:
        image = np.load('data/' + filepath + '-image.npy').astype('uint16')
        s1 = s1 + np.sum(image.flatten())
        s2 = s2 + np.sum(np.square(image.flatten()))
    normalization_params = {}
    normalization_params['mean'] = s1/N
    normalization_params['std'] = np.sqrt((N*s2-s1**2)/(N*(N-1))) 
    pickle.dump( normalization_params, open( crossvalidation_dir + '/normalization_params.p', "wb" ) )
        
    # Train
    training_generator = DataGeneratorTrain(partition['train'], normalization_params, params)
    validation_generator = DataGeneratorVal(partition['val'], normalization_params, params)
    
    history = model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    steps_per_epoch=int(len(partition['train'])/params['batch_size']),
                    validation_steps=len(partition['val']),
                    verbose=1,
                    epochs=params['epochs']-params_previous['epochs'])
    
    model.save(crossvalidation_dir + '/weights.h5')
    
    # Merge and save histories
    hist2 = history.history # history for the new epochs
    if hist1 == None: # No previous model
        hist = hist2
    elif hist2 == {}:
        hist = hist1
    else:
        hist = {}
        for key in hist1.keys():
            hist[key] = hist1[key] + hist2[key]
    pickle.dump( hist, open( crossvalidation_dir + '/history.p', "wb" ) )
    
    save_history(hist, crossvalidation_dir, params)
