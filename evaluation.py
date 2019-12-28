# Compute the prediction, the metrics (overlap: Dice, Jaccard, distance: maximum Hausdorff distance, mean Hausdorff distance, 95th percentile Hausdorff distance) and volumes.

from keras.models import load_model
import numpy as np
import os
from scipy import ndimage
import pickle
from sklearn.metrics import f1_score, confusion_matrix
from scipy.spatial.distance import directed_hausdorff, cdist
from models import *
from utils import *

image_size = [160, 160, 128]
spacing = [1.2, 1.2, 1.5]
organs_names = ['bladder', 'rectum', 'prostate']

def evaluate(filenames, src_dir, metric_types, gpu, save_predictions=False):
    # filenames: list of the filenames of the images fro which we want to compute the network prediction and and the performance criteria
    # src_dir: path of the folder containing the model weights and the normalization parameters. The predictions are saved in the folder src_dir/predictions
    # and the performance criteria are saved in the folder src_dir/performance_criteria
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    
    n_images = len(filenames)
    
    print('loading model...')
    
    model = load_model(src_dir + '/weights.h5', custom_objects={'dl': dl, 'b':b, 'r':r, 'p': p})
    print('done.')
    normalization_params = pickle.load( open( src_dir + '/normalization_params.p', "rb" ) )

    if not os.path.exists(src_dir + '/predictions'):
        os.makedirs(src_dir + '/predictions')
    if not os.path.exists(src_dir + '/metrics'):
        os.makedirs(src_dir + '/metrics')
        
    print('Progress:')
    n_patients = len(filenames)
    for patient_num in range(n_patients):
        filename = filenames[patient_num]
        if os.path.exists(src_dir + '/predictions/' + filename + '_prediction.npy'):
            #print('Loading previously saved prediction')
            prediction = np.load(src_dir + '/predictions/' + filename + '_prediction.npy')
        else:
            image = np.load('data/' + filename + '-image.npy')
            image = np.expand_dims(image, axis=0)
            image = np.expand_dims(image, axis=-1)
            image = (image-normalization_params['mean'])/normalization_params['std']
            prediction = model.predict(image)
            prediction = prediction[0,:,:,:,:]
            del image
            if save_predictions:
                np.save(src_dir + '/predictions/' + filename + '_prediction.npy', prediction)
        if len(metric_types)>0:
            n_organs = prediction.shape[-1]-1
            prediction = to_categorical(prediction)
            mask = np.load('data/' + filename + '-mask.npy')
            mask = mask[:,:,:,:-1]
            if 'overlap' in metric_types or 'distance' in metric_types:
                metrics = compute_metrics(mask, prediction, metric_types) 
                pickle.dump( metrics, open( src_dir + '/metrics/' + filename + '_metrics.p', "wb" ) )
            if 'volume' in metric_types:
                volumes = compute_volumes(prediction)
                filepath = src_dir + '/volumes'
                if not os.path.exists(filepath):
                    os.makedirs(filepath)
                np.save(src_dir + '/volumes/' + filename + '_volumes.npy', volumes)
        if 'overlap' in metric_types:
            print('|-- ' + str(patient_num) + ' ' + str(metrics['DSCs']))
        elif 'volume' in metric_types:
            print('|-- ' + str(patient_num) + ' ' + str(volumes))
        else:
            print('|-- ' + str(patient_num))

def compute_metrics(mask, prediction, metric_types):
    # mask: array of size (length, width, height, n_organs)
    # prediction: array of size (length, width, height, n_organs+1)
    # metric_types: list with one of the following elements: 'overlap', 'distance'.
    n_organs = mask.shape[-1]
    metrics = {}
    if 'overlap' in metric_types:
        metrics['DSCs'] = np.zeros(n_organs)
        metrics['JIs'] = np.zeros(n_organs)
        for organ_num in range(n_organs):
            if np.sum(mask[:,:,:,organ_num])==0:
                metrics['DSCs'][organ_num] = float('nan')
                metrics['DSCs'][organ_num] = float('nan')
            else:
                tn, fp, fn, tp = confusion_matrix(prediction[:,:,:,organ_num].flatten(), mask[:,:,:,organ_num].flatten()).ravel()
                metrics['DSCs'][organ_num] = 2*tp/(tp+fp+tp+fn)
                metrics['JIs'][organ_num] = tp/(fn+tp+fp)
    if 'distance' in metric_types:
        metrics['HDs'] = np.zeros(n_organs)
        metrics['HD95s'] = np.zeros(n_organs)
        metrics['HDmeans'] = np.zeros(n_organs)
        for organ_num in range(n_organs):
            if np.sum(mask[:,:,:,organ_num])==0:
                metrics['HDs'] = float('nan')
                metrics['HD95s'] = float('nan')
                metrics['HDmeans'][organ_num] = float('nan')
            else:
                mask_contours = mask2contours(mask[:,:,:,organ_num])
                pred_contours = mask2contours(prediction[:,:,:,organ_num])
                coord1 = np.argwhere(mask_contours)*np.asarray(spacing)
                coord2 = np.argwhere(pred_contours)*np.asarray(spacing)
                Y = cdist(coord1,coord2)
                if Y.shape[1]>0:
                    DAB = np.min(Y,axis=1)
                    DBA = np.min(Y,axis=0)
                    metrics['HDs'][organ_num] = np.max([np.max(DAB), np.max(DBA)])
                    metrics['HD95s'][organ_num] = np.mean([np.percentile(DAB, 95), np.percentile(DBA, 95)])
                    metrics['HDmeans'][organ_num] = np.mean([np.mean(DAB), np.mean(DBA)])
                else:
                    metrics['HDs'][organ_num] = None
                    metrics['HD95s'][organ_num] = None
                    metrics['HDmeans'][organ_num] = None
    return metrics

def compute_volumes(mask):
    # mask: array of size (length, width, height, n_organs)
    n_organs = mask.shape[-1]
    volumes = np.zeros(n_organs)                    
    for organ_num in range(n_organs):
        volumes[organ_num] = np.sum(mask[:,:,:,organ_num])*np.prod(spacing)
    return volumes