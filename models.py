from keras.models import Model, Sequential, load_model
from keras.layers import Input, concatenate, Conv3D, MaxPooling3D, Conv3DTranspose, Conv2D, MaxPooling2D, \
    Conv2DTranspose, Dropout, Flatten, Dense
import tensorflow as tf
from keras import backend as K
import numpy as np
from keras.optimizers import Adam
import keras
import os
from keras.callbacks import ModelCheckpoint, EarlyStopping
from math import log10, floor
import matplotlib.pyplot as plt
from scipy.misc import imfilter
from scipy import ndimage
import pickle
from data_generators import *
from keras import regularizers
from sklearn.metrics import f1_score, confusion_matrix
from scipy.spatial.distance import directed_hausdorff, cdist
from keras.utils import np_utils

image_size = [160, 160, 128]
spacing = [1.2, 1.2, 1.5]
organs_names = ['bladder', 'rectum', 'prostate']

def b(y_true, y_pred):
    sh = tf.shape(y_true)
    y_true_f = tf.transpose(tf.reshape(y_true[:,:,:,:,0], [sh[0], sh[1] * sh[2] * sh[3]]))
    y_pred_f = tf.transpose(tf.reshape(y_pred[:,:,:,:,0], [sh[0], sh[1] * sh[2] * sh[3]]))
    intersection = tf.multiply(y_true_f, y_pred_f)
    intersection = tf.reduce_sum(intersection, 0)
    card_y_true = tf.reduce_sum(y_true_f, 0)
    card_y_pred = tf.reduce_sum(y_pred_f, 0)
    dices = tf.div(2 * intersection, card_y_true + card_y_pred)
    return tf.reduce_mean(dices)

def r(y_true, y_pred):
    sh = tf.shape(y_true)
    y_true_f = tf.transpose(tf.reshape(y_true[:,:,:,:,1], [sh[0], sh[1] * sh[2] * sh[3]]))
    y_pred_f = tf.transpose(tf.reshape(y_pred[:,:,:,:,1], [sh[0], sh[1] * sh[2] * sh[3]]))
    intersection = tf.multiply(y_true_f, y_pred_f)
    intersection = tf.reduce_sum(intersection, 0)
    card_y_true = tf.reduce_sum(y_true_f, 0)
    card_y_pred = tf.reduce_sum(y_pred_f, 0)
    dices = tf.div(2 * intersection, card_y_true + card_y_pred)
    return tf.reduce_mean(dices)

def p(y_true, y_pred):
    sh = tf.shape(y_true)
    y_true_f = tf.transpose(tf.reshape(y_true[:,:,:,:,2], [sh[0], sh[1] * sh[2] * sh[3]]))
    y_pred_f = tf.transpose(tf.reshape(y_pred[:,:,:,:,2], [sh[0], sh[1] * sh[2] * sh[3]]))
    intersection = tf.multiply(y_true_f, y_pred_f)
    intersection = tf.reduce_sum(intersection, 0)
    card_y_true = tf.reduce_sum(y_true_f, 0)
    card_y_pred = tf.reduce_sum(y_pred_f, 0)
    dices = tf.div(2 * intersection, card_y_true + card_y_pred)
    return tf.reduce_mean(dices)


def gen_dice_loss(y_true, y_pred):
    sh = tf.shape(y_true)
    y_true_b = tf.transpose(tf.reshape(y_true[:,:,:,:,0], [sh[0], sh[1] * sh[2] * sh[3]]))
    y_pred_b = tf.transpose(tf.reshape(y_pred[:,:,:,:,0], [sh[0], sh[1] * sh[2] * sh[3]]))
    intersection_b = tf.multiply(y_true_b, y_pred_b)
    intersection_b = tf.reduce_sum(intersection_b, 0)
    card_y_true_b = tf.reduce_sum(y_true_b, 0)
    card_y_pred_b = tf.reduce_sum(y_pred_b, 0)
    w_b = 1/(card_y_true_b**2)
    
    y_true_r = tf.transpose(tf.reshape(y_true[:,:,:,:,1], [sh[0], sh[1] * sh[2] * sh[3]]))
    y_pred_r = tf.transpose(tf.reshape(y_pred[:,:,:,:,1], [sh[0], sh[1] * sh[2] * sh[3]]))
    intersection_r = tf.multiply(y_true_r, y_pred_r)
    intersection_r = tf.reduce_sum(intersection_r, 0)
    card_y_true_r = tf.reduce_sum(y_true_r, 0)
    card_y_pred_r = tf.reduce_sum(y_pred_r, 0)
    w_r = 1/(card_y_true_r**2)
    
    y_true_p = tf.transpose(tf.reshape(y_true[:,:,:,:,2], [sh[0], sh[1] * sh[2] * sh[3]]))
    y_pred_p = tf.transpose(tf.reshape(y_pred[:,:,:,:,2], [sh[0], sh[1] * sh[2] * sh[3]]))
    intersection_p = tf.multiply(y_true_p, y_pred_p)
    intersection_p = tf.reduce_sum(intersection_p, 0)
    card_y_true_p = tf.reduce_sum(y_true_p, 0)
    card_y_pred_p = tf.reduce_sum(y_pred_p, 0)
    w_p = 1/(card_y_true_p**2)
    
    dices = tf.div(2 * (w_b*intersection_b + w_r*intersection_r + w_p*intersection_p), 
                        w_b*(card_y_true_b + card_y_pred_b) + w_r*(card_y_true_r + card_y_pred_r) + w_p*(card_y_true_p + card_y_pred_p))
    return -dices
    
def dl(y_true, y_pred):
    return -0.33333333*tf.add(tf.add(b(y_true, y_pred), r(y_true, y_pred)), p(y_true, y_pred))

def unet_3d(params):
    n_organs = len(organs_names)
    nb_layers = len(params['feat_maps'])

    # Input layer
    inputs = Input(batch_shape=(None, *image_size, 1))

    # Encoding part
    skips = []
    x = inputs
    for block_num in range(nb_layers-1):
        nb_features = params['feat_maps'][block_num]
        x = Conv3D(nb_features, (3, 3, 3), activation='relu', padding='same')(x)
        x = Conv3D(nb_features, (3, 3, 3), activation='relu', padding='same')(x)
        skips.append(x)
        x = MaxPooling3D(pool_size=(2, 2, 2))(x)

    # Bottleneck
    nb_features = params['feat_maps'][-1]
    x = Conv3D(nb_features, (3, 3, 3), activation='relu', padding='same')(x)
    x = Conv3D(nb_features, (3, 3, 3), activation='relu', padding='same')(x)

    # Decoding part
    for block_num in reversed(range(nb_layers-1)):
        nb_features = params['feat_maps'][block_num]
        x = concatenate([Conv3DTranspose(nb_features, (2, 2, 2), strides=(2, 2, 2), padding='same')(x),
                         skips[block_num]], axis=4)
        x = Conv3D(nb_features, (3, 3, 3), activation='relu', padding='same')(x)
        x = Conv3D(nb_features, (3, 3, 3), activation='relu', padding='same')(x)

    # Output layer
    outputs = Conv3D(n_organs+1, (1, 1, 1), activation='softmax')(x)

    model = Model(inputs=[inputs], outputs=[outputs])
    if params['loss']=='categorical_crossentropy':
        model.compile(optimizer=Adam(params['lr']), loss='categorical_crossentropy', metrics=[b, r, p])
    elif params['loss']=='gen_dice_loss':
        model.compile(optimizer=Adam(params['lr']), loss=gen_dice_loss, metrics=[b, r, p])
    elif params['loss']=='dl':
        model.compile(optimizer=Adam(params['lr']), loss=dl, metrics=[b, r, p])
    return model
