# pelvis_segmentation

Get started by creating a folder named "data". In this folder, insert your images and masks in the "numpy array" format (one file per image and one file per masks). Images are expected to have a size (length, width, height), while masks are expcted to be (length, width, height, n_organs+1), where n_organs is the number of structures. E.g.,
If for example you have two image modalities CT and CBCT, please respect the following organization:  

data  
|-- CBCT-0-image.npy  
|-- CBCT-0-mask.npy  
|-- CBCT-1-image.npy  
|-- CBCT-1-mask.npy  
etc.  
|-- CT-0-image.npy  
|-- CT-0-mask.npy  
|-- CT-1-image.npy  
|-- CT-1-mask.npy  
etc.  

Finally, don't forget to update the beginning of each .py file of this repository with your image size, image spacing and the names of the structures to be segmented.

The file main.py shows an example of use with a 3-fold cross-validation using 63 CBCTs and 74 CTs.


