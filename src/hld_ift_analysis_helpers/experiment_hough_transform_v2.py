#python "D:/projects/HLD_parameter_determination/hld_ift_analysis_helpers/scripts/experiment_hough_transform_v2.py"
#
import sys
sys.path.append("D:/projects/HLD_parameter_determination/hld_ift_calc/src")
sys.path.append("D:/projects/HLD_parameter_determination/hld_ift_calc")
sys.path.append("D:/projects/HLD_parameter_determination/hld_ift_analysis_helpers/src")
#import pandas as pd
import re
import os
import time
import math
import functools
import itertools

#import copy
import numpy as np

from skimage.transform import hough_line, hough_line_peaks
from skimage.feature import canny
from skimage.draw import line as draw_line
from skimage import data
from skimage.io import imread,imsave
from skimage.color import rgb2gray

from skimage.filters import scharr

import matplotlib.pyplot as plt
from matplotlib import cm


from collect_files_folders import collect_files
from droplet_stats import needle_image, region_below_needle

from skimage.morphology import skeletonize as skeletonize_skim
from skimage.util import invert

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

from drop_recognition import (
                            contour_below_needle_tip_boundary,
                            #new_image_name,
                            needle_trace_distribution,
                            edges_needle_and_droplet,
                            connected_bits_below,
                            contour_below_needle,
                            needle_region_threshold,
                            autocrop_bin
                            ) 

#================================================================================
def create_fraction_to_intensity_fn(im, channels: list = [0], nbins = 256, a_range = None, mask = None):
    N = functools.reduce(lambda x, y: x * y, im.shape, 1) if mask is None else mask.sum() / mask.max() 
    im_min = 0 # im.min()
    im_max = 256  # im.max()
    range_to_use = a_range if a_range is not None else [im_min, im_max]
    #    hist = cv2.calcHist([img],[0],None,[256],[0,256])
    #accumulate
    a_hist = cv.calcHist([im], channels, mask, [nbins], range_to_use)
    print(a_hist)
    print(a_hist.shape)
    N2 = functools.reduce(lambda x,y: x+y[0],a_hist.tolist(), 0) 
    print(f'N2: {N2}')
    fraction_vals = itertools.accumulate(list(map(lambda x: x[0] / N, a_hist.tolist() )),
                                            lambda x,y: x + y)
    print(f'N: {N}')
    print(f'im.shape: {im.shape}')

    intensity_increment = (im_max - im_min) / (nbins - 1)
    intensity_vals = [im_min + x * intensity_increment for x in range(nbins) ]
    for intensity, fraction in zip(intensity_vals, fraction_vals):
        print(f'{fraction}, {intensity}')

    def temp(fr):
        if fr >= 1:
            return intensity_vals[len(intensity_vals) - 1]
        elif fr <=0:
            return intensity_vals[0]
        else:
            index = [index for index,value in enumerate(fraction_vals) if value > fr ]
            print(index)
            index = index[0]
            return interpolate(fr, fraction_vals[index - 1], fraction_vals[index], intensity_vals[index - 1], intensity_vals[index])

    return temp

def interpolate(x_val, x1, x2, y1, y2):
    return y1 + (y2 - y1) / (x2 - x1) * (x_val - x1)

#================================================================================


start = time.time()

def temp_data_output_exp_name(path):
    return os.path.join("D:/temp_data/output", os.path.split(path)[1])
root = "D:/temp_data/exp_2025-02-24_001_20g_BrijL4_C7-C16"
root_out = "D:/temp_data/output/exp_2025-02-24_001_20g_BrijL4_C7-C16"

root = "D:/temp_data/exp_2025-03-26_06g_BrijL4_C7C16_001"
root_out = "D:/temp_data/output/exp_2025-03-26_06g_BrijL4_C7C16_001"

root = r'\\huckdfs-srv.science.ru.nl\huckdfs\RobotLab\Storage-Miscellaneous\aigars\temp\AOT_IB-45_C7C16_NaCl\exp_2025-04-07_05g_AOT_C7C16_001'
root_out = temp_data_output_exp_name(root)
print(root_out)


#jpgs = collect_files(root, "[0-9]{5}.jpg$") #"conc_[0-9.]{7}[\/].*[0-9]{5}.jpg$")
#jpgs = collect_files(root, "conc_[0-9.]{7}[\\\\].*[0-9]{5}.jpg$")
jpgs = [
        r"\\huckdfs-srv.science.ru.nl\huckdfs\RobotLab\Storage-Miscellaneous\aigars\temp\AOT_IB-45_C7C16_NaCl\exp_2025-04-07_05g_AOT_C7C16_001\scan_001\conc_0.50000\00012.jpg"
        ]




for i, im in enumerate(jpgs):
    if i < 5:
        print(f'{i: >5d}: {im}')


def new_image_name(p, root, root_out):
    x = os.path.split(p)
    if x[0] == root:
        return os.path.join(root_out, x[1])
    else:
        return os.path.join(new_image_name(x[0], root, root_out), x[1])

for i,p in enumerate(jpgs):
    #if i < 1450 and i > 1448: #10: #2500: # and i >= 1500:
    if i > -1:
        output_img_path = new_image_name(p, root, root_out)
        print(f'{i: >3d}: {output_img_path}')
        #save_edges_needle_and_droplet(p, output_img_path)

        #>edge_distribution_path = derive_from_new(p, root, root_out, "a_trace_distr_")
        #>print(f'{i: >3d}: {edge_distribution_path}')
        #>needle_trace_distribution_plot(output_img_path, edge_distribution_path)        

        value = 70
        erode = 0
        dilate = 0
        fraction = 0.05
        one_more = True
        show_image = True
        while one_more:
            im = imread(p)
            im = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
            fract_to_intsty = create_fraction_to_intensity_fn(im, [0])
            ####---->for a_fr in [ x / 20 for x in range(21) ]:
            ####---->    print(f'fract_to_intsty({a_fr}): {fract_to_intsty(a_fr)}')
            yyy = needle_region_threshold(im, fraction = fraction, show_image = show_image)
            print(im.shape)
            edges_needle_and_droplet(yyy, show_image = show_image, threshold_low = value, erode = erode, dilate = dilate)

            #\\\>>>x = contour_below_needle(im, show_image = show_image, threshold_low = value, dilate = dilate, erode = erode)
            #\\\>>>#plt.imshow(x)
            #\\\>>>#plt.title("main loop: contour below")
            #\\\>>>#plt.show()
            #\\\>>>print(x.dtype)
            #\\\>>>os.makedirs(os.path.split(output_img_path)[0], exist_ok = True)
            #\\\>>>print(output_img_path)
            #\\\>>>cv.imwrite(output_img_path, autocrop_bin(x).astype(np.uint8))    
            #\\\>>>print(f'done writing: ... {output_img_path}')
            #\\\>>>print(f'show_image: {show_image}')
            if show_image:
                iii = input('threshold value or `q`')
                one_more = not iii == 'q'
                if one_more:
                    cmpnnts = iii.split(",")
                    value = int(cmpnnts[0])
                    dilate = int(cmpnnts[1])
                    erode = int(cmpnnts[2])
                    fraction = float(cmpnnts[3])
            else:
                one_more = False

        #===> |edges = edges_needle_and_droplet(im) 

        #===> |#--> im = 255 * (im > 127) #saving to jpg creates artifacts that can be removed by thresholding image!!!
        #===> |#>plt.imshow(im)
        #===> |#>plt.title("1")
        #===> |#>plt.show()
        #===> |x_sec,max_val,max_val_indexes = needle_trace_distribution(edges)
        #===> |#>plt.imshow(im)
        #===> |#>plt.title("2")
        #===> |#>plt.show()
        #===> |print(f'main bit: max_val_indexes: {max_val_indexes}')

        #===> |#__>llcb = connected_bits_below(im, max_val_indexes[0])
        #===> |#__>contour_path = derive_from_new(p, root, root_out, "a_drop_contour_")
        #===> |#__>#>plt.imshow(llcb)
        #===> |#__>#>plt.title("3")
        #===> |#__>#>plt.show()
        #===> |#__>imsave(contour_path, llcb)
        #===> |cntr = connected_bits_below(edges, max_val_indexes[0])
        #===> |iim = contour_below_needle_tip_boundary(cntr, max_val_indexes[0], max_theta = 90, angles = 360, delta_px = 10)
        #===> |plt.imshow(iim)
        #===> |plt.title("contour below")
        #===> |plt.show()
        print(p)


print(f'duration: {time.time() - start}')

