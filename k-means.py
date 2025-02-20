# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 12:13:58 2020

@author: Anna
"""


import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mp

from IPython.display import display

import numpy as np

from scipy import ndimage, misc,signal

from sklearn.metrics import silhouette_score
from sklearn import cluster, datasets, mixture
from sklearn.neighbors import kneighbors_graph
from sklearn.cluster import DBSCAN, KMeans

from skimage import morphology
from skimage.segmentation import flood, flood_fill

import glob

import cv2

import PIL
from PIL import Image, ImageDraw, ImageColor, ImageFilter, ImageEnhance

import tifffile as tiff
from tifffile import imsave

#%%
path_1 = r'C:/.../raw/' #FOLDER WITH RAW/INITIAL IMAGES
all_files = glob.glob(path_1 + "/*.tif")

path_2 = r'C:/Users/.../cluster/' #FOLDER WITH SEGMENTED IMAGES OF ALL PHASES

path_3 = r'C:/Users/.../pores/' #FOLDER WITH SEGMENTED IMAGES OF PORES (DARKEST PHASES)

path_4 = r'C:/Users/.../UH/' #FOLDER WITH SEGMENTED IMAGES OF BRIGHTEST PHASES

  
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 30}

mp.rc('font', **font)
mp.rcParams.update({'font.size': 30})

def what_y_if_x(input, y,x):
    return y[x.searchsorted(input, 'left')]

#%%
for i in range(1,len(all_files)+1):
    
    OrigImage = Image.open(path_1+"S_"+str(i)+".jpg")
    OrigImage = np.array(OrigImage)
    OrigImage = OrigImage[200:2500, 500:3800] 
    OrigImage= ndimage.median_filter(OrigImage, size=5)
    img_row = OrigImage.reshape(OrigImage.shape[0]*OrigImage.shape[1],-1)

        
#%%CLUSTERING ERROR ESTIMATION WITH GRAY-LEVEL VALUES OF THE PIXELS
#NO NEED TO RUN WITH EVERY IMAGE, THEREFORE, UNCOMMENT WHEN NEEDED TO ESTIMATE NUMBER OF CLUSTERS


# k=8## MANUALLY CHOSEN NUMBER OF CLUSTERS
# m_im = img_row.shape[0]
# err_dif_clus_nr=np.zeros((k,1))    # Array for storing clustering errors
# list_clus=list(range(1,k+1))
# k_i=len(list_clus)
# cluster_assign= np.zeros((k_i, m_im), dtype=np.int32)  # Array for storing clustering assignments
# it=0
# for cl in list_clus:
#     kmeans_im = KMeans(n_clusters = cl, max_iter = 100).fit(img_row)
#     cluster_labels = kmeans_im.labels_
#     cluster_assign[it]= cluster_labels
#     inertia_end = kmeans_im.inertia_
#     err_dif_clus_nr[it] = (1/m_im)*inertia_end
#     it+=1

# print(f'Clustering errors: \n{err_dif_clus_nr}')  
    
    
#%% SEGMENTATION OF ALL PHASES
    
    k=5 #MANUALLY DEFINED BASED ON CLUSTERING ERROR (PREVIOUS CELL)
    kmeans_img = KMeans(n_clusters=k, max_iter=100).fit(img_row) #THE NUMBER OF MAXIMUM ITINERARIES CAN BE CHANGED, BUT 100 IS THE OPTIMUM PRECISION AND TIME-WISE
    clus_mean=kmeans_img.cluster_centers_
    clus_indicies=kmeans_img.labels_
    clus_error=kmeans_img.inertia_
    
    img2show =  clus_mean[clus_indicies]
    clus_img = img2show.reshape(OrigImage.shape[0], OrigImage.shape[1])
    #plt.imshow(clus_img, cmap="gray"), plt.axis('off') #UNCOMMENT THE LINE IF WANT TO CHECK THE RESULTED IMAGE IN "PLOT" TAB
    
    cv2.imwrite(path_2+"cluster_"+str(i)+".png", clus_img) 

#%% SEGMENTATION OF ONLY DARKEST PHASES

        
    im_bin_p = (clus_img <= min(kmeans_img.cluster_centers_))
    im_bin_p = morphology.binary_opening(im_bin_p)
    im_bin_p=np.invert(im_bin_p) 
    plt.imshow(im_bin_p, cmap="gray"), plt.axis('off') 
    im_bin_p = Image.fromarray(im_bin_p)
        
    im_bin_p.save(path_3+"pores_"+str(i)+".tif")
    
#%% SEGMENTATION OF ONLY BRIGHTEST PHASES
  
    im_bin = (clus_img >= max(kmeans_img.cluster_centers_))
    im_bin =flood_fill(im_bin, (1000, 1000), 0)
    im_bin = morphology.binary_opening(im_bin)
    im_bin = morphology.remove_small_objects(im_bin, 20)
    im_bin=np.invert(im_bin) 
    im_bin = Image.fromarray(im_bin)
    #plt.imshow(im_bin, cmap="gray"), plt.axis('off') #UNCOMMENT THE LINE IF WANT TO CHECK THE RESULTED IMAGE IN "PLOT" TAB
      
    im_bin.save(path_4+"UH_"+str(i)+".tif")
 


    
