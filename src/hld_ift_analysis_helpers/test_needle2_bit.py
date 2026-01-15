#python "D:/projects/HLD_parameter_determination/hld_ift_analysis_helpers/src/hld_ift_analysis_helpers/test_needle2_bit.py"
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

images = [
        "//huckdfs-srv.science.ru.nl/huckdfs/RobotLab/Storage-Miscellaneous/aigars/temp/Ecosurf_EH_3_C7_C16_OIW/exp_2025-08-07_05.00g_Ecosurf_EH_3_OIW_C7_C16_NaCl_9pts_001/scan_001/conc_0.00000_autofocus_needle_detection_image/00000.jpg", 
        "//huckdfs-srv.science.ru.nl/huckdfs/RobotLab/Storage-Miscellaneous/aigars/temp/Ecosurf_EH_3_C7_C16_OIW/exp_2025-08-07_05.00g_Ecosurf_EH_3_OIW_C7_C16_NaCl_9pts_001/scan_001/conc_1.00000_autofocus_needle_detection_image/00000.jpg"
        ]

def print_im_props(im, label):
    print(f'---------------------------- im props: {label}')
    print(type(im))
    print(im)
    print(im.shape)
    print(f'---------------------------- im props: {label}')

#
from skimage import data, util
from skimage.measure import label, regionprops
from skimage import data, filters, measure, morphology

p = images[0]

im = imread(p, as_gray = True)
#im = cv.cvtColor(im, cv.COLOR_BGR2GRAY)

print_im_props(im, "im")

im_max = im.max()
print(f'im_max: {im_max}')

im2 = im_max - im
    
threshold = filters.threshold_otsu(im2)
print_im_props(threshold, "threshold")

mask = im2 > threshold
print_im_props(mask, "mask")

mask = morphology.remove_small_objects(mask, 50)
mask = morphology.remove_small_holes(mask, 50)

labels = measure.label(mask)


def show_image(im, a_title = "............"):
    plt.imshow(im)
    plt.title(a_title)
    plt.show()
    

show_image(labels, "labels")

#def index_list(rm):
#    return rm.where(
#def region_min_x(rm):
#def region_max_x(rm):
#def region_min_y(rm):
#def region_max_y(rm):

props = regionprops(labels)

def min_x(coords):
    return coords.min(axis = 0)

def pixelcount(regionmask):
    return np.sum(regionmask)

def rel_size(regionmask):
    shp = im.shape
    return np.sum(regionmask) /  (shp[0] * shp[1])

def extrema(coords):
    mins = np.min(coords, axis = 0)
    maxs = np.max(coords, axis = 0)
    return dict(min_x = mins[1], max_x = maxs[1], min_y = mins[0], max_y = maxs[0])


props = regionprops(labels, extra_properties=(pixelcount, rel_size,))

i_area_lst = list(map(lambda x: [x[0], -x[1].area], enumerate(props)))
area_sorted = sorted(i_area_lst, key = lambda x: x[1])

def region_from_top_to_bottom(roi_prop):
    """
        fn checks if region spans across whole y range
        roi_prop: a region created by regionprops fn
    """
    print("region_from_top_to_bottom___st")
    ext = extrema(roi_prop.coords)
    print("region_from_top_to_bottom___end")
    return ext["min_y"] == 0 and ext["max_y"] == im.shape[0] - 1

area_sorted_clup = list(filter(lambda x: not region_from_top_to_bottom(props[x[0]]), area_sorted))

flipped = len(area_sorted_clup) < len(area_sorted)

region_prop = props[area_sorted_clup[0][0]]

width = 150

coords = region_prop.coords
extr_val = extrema(coords)
start = extr_val["min_x"]
end = extr_val["max_x"]
center = int((start + end) / 2)
width_raw = end - start
width_report = width_raw if width <= 0 else width
half = int( width_report / 2 )
start = center - half if center - half >= 0 else 0
#    return {"start": start, "width": width_report, "center": center}
print({"start": start, "width": width_report, "center": center})





for i,p in enumerate(props):
    print(f' ------------------------- {i}')
    print(f' area: {p.area}')
    print(f' coords: {p.coords}')
    print(f' pixelcount: {p.pixelcount}')
    print(f' rel_size: {p.rel_size}')

    coord = p.coords
    ext = extrema(coord)
    print(f' min_x: {ext["min_x"]}')
    print(f' min_y: {ext["min_y"]}')
    print(f' max_x: {ext["max_x"]}')
    print(f' max_y: {ext["max_y"]}')
   

   #print(f' rel_size_region: {p.rel_size_region}')



