import numpy as np
import os
import matplotlib.pyplot as plt
#from scipy import ndimage
#from skimage.color import gray2rgb
import pickle
from sklearn.metrics import confusion_matrix
from scipy.spatial.distance import directed_hausdorff, cdist
from scipy.misc import imfilter
import ipyvolume as ipv
from models import *

image_size = [160, 160, 128]
spacing = [1.2, 1.2, 1.5]
organs_names = ['bladder', 'rectum', 'prostate']

rgb_dark = np.array([[255,0,0],
                     [0,128,0],
                     [0,0,255]])
rgb_light = np.array([[255,0,255],
                     [0,255,0],
                     [0,255,255]])

def to_categorical(prediction):
    n_organs = prediction.shape[-1]-1
    prediction_thr = np.argmax(prediction, axis=-1)
    prediction_new = np.zeros((*image_size,n_organs))
    for organ_num in range(n_organs):
        prediction_new[:,:,:,organ_num] = (prediction_thr==organ_num)
    return prediction_new

def save_history(hist, dest_dir, params):
    n_organs = len(organs_names)
    
    x = range(1, len(hist['loss'])+1)
    plt.figure(figsize=(12, 4+n_organs*4))
    plt.subplot(n_organs+1, 1, 1)
    plt.plot(x, hist['loss'], 'o-', label='Training')
    plt.plot(x, hist['val_loss'], 'o-', label='Validation')
    plt.legend(loc='upper left')
    plt.ylabel('Loss')
    plt.grid(True)
    
    for organ_num in range(n_organs):
        plt.subplot(n_organs+1, 1, 1+organ_num+1)
        plt.plot(x, hist[organs_names[organ_num][0]], 'o-')
        plt.plot(x, hist['val_'+organs_names[organ_num][0]], 'o-')
        plt.ylabel('Dice '+organs_names[organ_num])
        plt.grid(True)
    
    plt.savefig(dest_dir + '/learning_curves.png')

def params2name(params):
    results_name = ''
    for key in params.keys():
        results_name = results_name + key + '_' + str(params[key]) +'_'
    results_name = results_name[:-1]
    return results_name
        
# Visualization

def image_with_2contours(image, mask1, mask2, color1=rgb_dark, color2=rgb_light):
    vmin = -1000
    vmax = 3000
    output = ((image - vmin) * 255 / (vmax - vmin)).astype(np.uint8)
    mask1 = imfilter(mask1*255, 'find_edges')
    #mask1 = mask1.filter(ImageFilter.FIND_EDGES)
    mask2 = imfilter(mask2*255, 'find_edges')
    output = gray2rgb(output)
    output[mask1 > 0] = np.array(color1)
    output[mask2 > 0] = np.array(color2)
    return output

def show_slices(image, masks, *argv):
    sh = image.shape
    n_slices = sh[2]
    n_lines = int(n_slices/16)
    im = np.zeros((sh[0]*n_lines,sh[1]*16,3))
    plt.figure(figsize=(40,20))
    if len(argv)==1 or len(argv)==0:
        for s in range(n_slices):
            output = image_with_2contours(image[:,:,s], masks[:,:,s,0], masks[:,:,s,1], [255,255,0], [255,255,0])
            line = (s//16)*sh[0]
            col = (s%16)*sh[0]
            im[line:(line+sh[0]),col:(col+sh[0]),:] = output
        im = ((im - np.min(im)) * 255 / (np.max(im) - np.min(im))).astype(np.uint8)
        frame1 = plt.gca()
        frame1.axes.get_xaxis().set_visible(False)
        frame1.axes.get_yaxis().set_visible(False)
        plt.imshow(im)  
    
        filepath = argv[0]
        print(filepath)
        plt.show()
        plt.close()
    else:
        masks2 = argv[0]
        info = argv[1]
        
        for s in range(n_slices):
            output = image_with_4contours(image[:,:,s], masks[:,:,s,0], masks[:,:,s,1], masks2[:,:,s,0], masks2[:,:,s,1], [255,255,0], [255,0,0])
            line = (s//16)*sh[0]
            col = (s%16)*sh[0]
            im[line:(line+sh[0]),col:(col+sh[0]),:] = output
        im = ((im - np.min(im)) * 255 / (np.max(im) - np.min(im))).astype(np.uint8)
        frame1 = plt.gca()
        frame1.axes.get_xaxis().set_visible(False)
        frame1.axes.get_yaxis().set_visible(False)
        plt.imshow(im)  
        print(info)
        plt.show()
        plt.close()
    
def image_with_4contours(image, true1, true2, pred1, pred2, color1, color2):
    output = image
    true1 = imfilter(true1*255, 'find_edges')
    true2 = imfilter(true2*255, 'find_edges')
    pred1 = imfilter(pred1*255, 'find_edges')
    pred2 = imfilter(pred2*255, 'find_edges')
    output = gray2rgb(output)
    output[true1 > 0] = np.array(color1)
    output[true2 > 0] = np.array(color1)
    output[pred1 > 0] = np.array(color2)
    output[pred2 > 0] = np.array(color2)
    return output

def image_with_contours_gen(image, bm1, bm2, color1=rgb_dark, color2=rgb_light):
    output = image
    output = gray2rgb(output)
    n_contours = bm1.shape[2]
    for contour_num in range(n_contours):
        edges = imfilter(bm1[:,:,contour_num]*255, 'find_edges')
        output[edges>0] = np.array(color1[contour_num,:])
    for contour_num in range(n_contours):
        edges = imfilter(bm2[:,:,contour_num]*255, 'find_edges')
        output[edges>0] = np.array(color2[contour_num,:])
    return output

def image_with_contours_new(image, bm1, bm2, color1=rgb_dark, color2=rgb_light):
    output = image
    output = gray2rgb(output)
    n_contours = bm1.shape[2]
    for contour_num in range(n_contours):
        edges = imfilter(bm1[:,:,contour_num]*255, 'find_edges')
        output[edges>0] = np.array(color1[contour_num,:])
    for contour_num in range(n_contours):
        edges = imfilter(bm2[:,:,contour_num]*255, 'find_edges')
        output[edges>0] = np.array(color2[contour_num,:])
    return output

def show_patient(src_dir, patient_name, view):
    if view=='2d':
        show_patient_2d(src_dir, patient_name)
    elif view=='3d':
        show_patient_3d(src_dir, patient_name)
    else:
        slice_num = int(view)
        image = np.load('data/' + patient_name + '-image.npy')
        sh = image.shape
        bm1 = np.load('data/' + patient_name + '-mask.npy')
        bm1 = bm1[:,:,:,:-1]
        prediction = np.load(src_dir + '/predictions/' + patient_name + '_prediction.npy')
        bm2 = to_categorical(prediction)
        output = image_with_contours_gen(image[:,:,slice_num], bm1[:,:,slice_num,:], bm2[:,:,slice_num,:])
        print(patient_name)
        plt.imshow(output)
        
def show_slices_gen_large(image, bm1, bm2, col1=rgb_dark, col2=rgb_light, info='', filename=None):
    sh = image.shape
    n_slices = sh[2]
    im = np.zeros((192*10,192*16,3))
    plt.figure(figsize=(40,20))
    for s in range(n_slices):
        output = image_with_contours_gen(image[:,:,s], bm1[:,:,s,:], bm2[:,:,s,:], col1, col2)
        line = (s//16)*sh[0]
        col = (s%16)*sh[0]
        im[line:(line+sh[0]),col:(col+sh[0]),:] = output
    im = ((im - np.min(im)) * 255 / (np.max(im) - np.min(im))).astype(np.uint8)
    frame1 = plt.gca()
    frame1.axes.get_xaxis().set_visible(False)
    frame1.axes.get_yaxis().set_visible(False)
    plt.imshow(im)
    print(info)
    plt.show()
    if filename != None:
        plt.imsave(filename, im, dpi=1000)
    plt.close()
    
def show_slices_gen(image, bm1, bm2=np.zeros((160,160,128,3)), col1=rgb_dark, col2=rgb_light, info='', filename=None):
    sh = image.shape
    n_slices = sh[2]
    im = np.zeros((160*8,160*16,3))
    plt.figure(figsize=(40,20))
    for s in range(n_slices):
        output = image_with_contours_gen(image[:,:,s], bm1[:,:,s,:], bm2[:,:,s,:], col1, col2)
        line = (s//16)*sh[0]
        col = (s%16)*sh[0]
        im[line:(line+sh[0]),col:(col+sh[0]),:] = output
    im = ((im - np.min(im)) * 255 / (np.max(im) - np.min(im))).astype(np.uint8)
    frame1 = plt.gca()
    frame1.axes.get_xaxis().set_visible(False)
    frame1.axes.get_yaxis().set_visible(False)
    plt.imshow(im)
    print(info)
    plt.show()
    if filename != None:
        plt.imsave(filename, im, dpi=1000)
    plt.close()
    
def show_patient_2d(src_dir, filename):
    image = np.load('data/' + filename + '-image.npy')
    sh = image.shape
    mask = np.load('data/' + filename + '-mask.npy')
    mask = mask[:,:,:,:-1]
    prediction = np.load(src_dir + '/predictions/' + filename + '_prediction.npy')
    n_organs = prediction.shape[-1]-1
    prediction = np.argmax(prediction, axis=-1)
    mask_pred = np.zeros((sh[0],sh[1],sh[2],n_organs))
    for organ_num in range(n_organs):
        mask_pred[:,:,:,organ_num] = (prediction==organ_num)
    col1 = np.array([[255,0,0],
                     [0,128,0],
                     [0,0,255]])
    col2 = np.array([[255,0,255],
                     [0,255,0],
                     [0,255,255]])
    show_slices_gen(image, mask, mask_pred, col1, col2, filename, filename)
    
def mask2contours(mask):
    sh = mask.shape
    contours = np.zeros((sh[0],sh[1],sh[2]))
    n_slices = sh[2]
    
    for s in range(n_slices):
        if s>0 and np.sum(mask[:,:,s-1].flatten())==0:
            contours[:,:,s] = mask[:,:,s]
        
        elif s<(n_slices-1) and np.sum(mask[:,:,s+1].flatten())==0:
            contours[:,:,s] = mask[:,:,s]
        else:
            diff = np.abs(mask[:,:,s]*1 - mask[:,:,s-1]*1)>0
            imf = imfilter(mask[:,:,s].astype('int'), 'find_edges')
            if np.sum(diff.flatten()>0):
                contours[:,:,s] = (diff+imf)>0
            else:
                contours[:, :, s] = imf
    return contours

def show_patient_3d(src_dir, patient_name):
    filepath = 'data/' + patient_name + '-mask.npy'
    mask = np.load(filepath)
    fig = ipv.figure()
    colors = ['red', 'green', 'blue']
    n_organs = 3
    contours = {}
    for organ_num in range(n_organs):
        contours = mask2contours(mask[:,:,:,organ_num])
        coord = np.argwhere(contours)
        n_points = coord.shape[0]
        x, y, z = np.random.normal(0,100,(3,n_points))
        for i in range(n_points):
            x[i] = coord[i,0]
            y[i] = coord[i,1]
            z[i] = coord[i,2]
        if n_points>0:
            scatter = ipv.scatter(x, y, z, size=1, marker='sphere', color=colors[organ_num])
    ipv.show()
    
    if src_dir != None:
        prediction = np.load(src_dir + '/predictions/' + patient_name + '_prediction.npy')
        prediction_thr = np.argmax(prediction, axis=-1)
        
        fig = ipv.figure()
        colors = ['magenta', 'lime', 'cyan']
        n_organs = 3
        contours = {}
        for organ_num in range(n_organs):
            mask = (prediction_thr==organ_num)
            contours = mask2contours(mask)
            coord = np.argwhere(contours)
            n_points = coord.shape[0]
            x, y, z = np.random.normal(0,100,(3,n_points))
            for i in range(n_points):
                x[i] = coord[i,0]
                y[i] = coord[i,1]
                z[i] = coord[i,2]
            if n_points>0:
                scatter = ipv.scatter(x, y, z, size=1, marker='sphere', color=colors[organ_num])
        ipv.show()


