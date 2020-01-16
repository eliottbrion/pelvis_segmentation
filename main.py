import os
import numpy as np
from utils import *
from models import *
import pickle
from random import randint
import sys
from data_generators import DataGeneratorTrain, DataGeneratorVal
from training import *

gpu = 0

# === Training ===

 params = {'lr':             1e-4,
              'batch_size': 2,
              'epochs':             150,
              'loss': 'dl',
              'feat_maps': [16, 32, 64, 128, 256, 512],
              'model': 'unet_3d',
               }

results_dir = 'results/nCTs_74_nCBCTs_42'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

inds = list(range(63))
for fold_num in range(3):
    inds = np.roll(inds,21*fold_num)
    previous_dir = None # The training starts from scratch (and not from a previously pre-trained network
    partition = {'train':   ['CBCT-' + str(inds[i]) for i in range(42)] + ['CBCT-' + str(inds[i]) for i in range(74)],
             'val':       ['CBCT-' + str(inds[i]) for i in np.arange(42,63)]}
    train(partition, previous_dir, gpu, results_dir, params)
    
# === Evaluation ===

metric_types = ['overlap', 'distance']

for fold_num in range(3):
    inds = list(range(63))
    inds = np.roll(inds,21*fold_num)
    filenames = ['CBCT-' + str(inds[i]) for i in np.arange(42,63)]
    src_dir = results_dir + '/' + params2name(params) + '/' + filenames[0]
    evaluate(filenames, src_dir, metric_types, gpu)
    
