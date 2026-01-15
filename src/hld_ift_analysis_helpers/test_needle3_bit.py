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
import matplotlib.patches as mpatches

#>import numpy as np
from skimage import util, filters, color #data, util, filters, color
from skimage.measure import label, regionprops
from skimage.segmentation import watershed
from skimage.filters import threshold_otsu
from skimage.color import label2rgb
#>import matplotlib.pyplot as plt

def needle_alternative(im, width = 0, debug = False):
    # threshold raw image
    # find regions
    # find largest region
    #   if largest region spans over height of image ==> inverted scan else ==> normal scan
    # if normal_scan: 
    #       give larges region as needle
    # else 
    #       give second largest region as needle

    #========================
    # OTSU thresholding
    image = im #data.camera()
    thresh = threshold_otsu(image)
    binary = image < thresh
    
    # label image regions
    label_image = label(binary)
   
    region_properties = regionprops(label_image) 

    reg_area = list(map(lambda x: -x.area, region_properties))
    index_by_area_size = np.array(reg_area).argsort()

    # ---- roi spans all rows???? ----
    largest_region = region_properties[index_by_area_size[0]]
    row1,col1,row2,col2 = largest_region.bbox

    scan_type = "inverse" if row2 - row1 == binary.shape[0] else "direct"
    roi_index = 1 if row2 - row1 == binary.shape[0] else 0

    region = region_properties[index_by_area_size[roi_index]]
    minr, minc, maxr, maxc = region.bbox

    if debug:
        # watershed segmentation
        
        #>coins = data.coins()
        edges = filters.sobel(binary)
        
        grid = util.regular_grid(binary.shape, n_points=20)
        
        seeds = np.zeros(binary.shape, dtype=int)
        seeds[grid] = np.arange(seeds[grid].size).reshape(seeds[grid].shape) + 1
        
        w0 = watershed(edges, seeds)
        w1 = watershed(edges, seeds, compactness=0.01)
        
        fig, (ax0, ax1) = plt.subplots(1, 2)
        
        ax0.imshow(color.label2rgb(w0, binary, bg_label=-1))
        ax0.set_title('Classical watershed')
        
        #111> ax1.imshow(color.label2rgb(w1, binary, bg_label=-1))
        #111> ax1.set_title('Compact watershed')
        #111> 

        #000> ax1.imshow(binary)
        #000> ax1.set_title('Compact watershed')
        #000> 

        # to make the background transparent, pass the value of `bg_label`,
        # and leave `bg_color` as `None` and `kind` as `overlay`
        image_label_overlay = label2rgb(label_image, image=binary, bg_label=0)

        ax1.imshow(image_label_overlay)
        ax1.set_title('Compact watershed')

        print(f'row1: {row1}, row2: {row2}, col1: {col1}, col2: {col2}')
        print(binary.shape)

        rect = mpatches.Rectangle(
            (minc, minr),
            maxc - minc,
            maxr - minr,
            fill=False,
            edgecolor='red',
            linewidth=2,
        )
        ax1.add_patch(rect)

        print(f'row1: {minr}, row2: {maxr}, col1: {minc}, col2: {maxc}')
        #for region in region_properties:
        #    # take regions with large enough areas
        #    if region.area >= 100:
        #        # draw rectangle around segmented coins
        #        minr, minc, maxr, maxc = region.bbox
        #        rect = mpatches.Rectangle(
        #            (minc, minr),
        #            maxc - minc,
        #            maxr - minr,
        #            fill=False,
        #            edgecolor='red',
        #            linewidth=2,
        #        )
        #        ax1.add_patch(rect)

        plt.show()
    
    center_raw = (minc + maxc) // 2
    width_raw = maxc - minc + 1
    width_report = width_raw if width == 0 else width
    half = width_report // 2
    start_raw = center_raw - half + 1
    start = start_raw if start_raw >= 0 else 0
    center_report = start + half

    if debug:
        print(f'original columns: col1: {minc}, col2: {maxc}')
        print(f'reported columns: col1: {start}, col2: {start + width_report - 1}')
        print(f'differences: min: {minc - start}, max: {start + width_report - 1 - maxc}')

    return {"start": start, "width": width_report, "center": center_report, "scan_type": scan_type}

    

images = [
        "//huckdfs-srv.science.ru.nl/huckdfs/RobotLab/Storage-Miscellaneous/aigars/temp/HLD_scan/Genapol_EP_2424_C7_C16/exp_2025-11-25_20.00g_Genapol_EP_2424_C7_C16_NaCl_001/scan_002/conc_0.00000/00000.jpg",
        "//huckdfs-srv.science.ru.nl/huckdfs/RobotLab/Storage-Miscellaneous/aigars/temp/HLD_scan/Ecosurf_EH_3_C7_C16_OIW/exp_2025-08-07_05.00g_Ecosurf_EH_3_OIW_C7_C16_NaCl_9pts_001/scan_001/conc_0.00000_autofocus_needle_detection_image/00000.jpg", 
        "//huckdfs-srv.science.ru.nl/huckdfs/RobotLab/Storage-Miscellaneous/aigars/temp/HLD_scan/Ecosurf_EH_3_C7_C16_OIW/exp_2025-08-07_05.00g_Ecosurf_EH_3_OIW_C7_C16_NaCl_9pts_001/scan_001/conc_1.00000_autofocus_needle_detection_image/00000.jpg"

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

def test(width = 0):
    print(f'----------{width}----------')
    print(needle_alternative(im, width = width, debug = True))
    print("")

test(0)
test(100)
test(101)
test(200)
test(201)

#______ print_im_props(im, "im")
#______ 
#______ im_max = im.max()
#______ print(f'im_max: {im_max}')
#______ 
#______ im2 = im_max - im
#______     
#______ threshold = filters.threshold_otsu(im2)
#______ print_im_props(threshold, "threshold")
#______ 
#______ mask = im2 > threshold
#______ print_im_props(mask, "mask")
#______ 
#______ mask = morphology.remove_small_objects(mask, 50)
#______ mask = morphology.remove_small_holes(mask, 50)
#______ 
#______ labels = measure.label(mask)
#______ 
#______ 
#______ def show_image(im, a_title = "............"):
#______     plt.imshow(im)
#______     plt.title(a_title)
#______     plt.show()
#______     
#______ 
#______ show_image(labels, "labels")
#______ 
#______ #def index_list(rm):
#______ #    return rm.where(
#______ #def region_min_x(rm):
#______ #def region_max_x(rm):
#______ #def region_min_y(rm):
#______ #def region_max_y(rm):
#______ 
#______ props = regionprops(labels)
#______ 
#______ def min_x(coords):
#______     return coords.min(axis = 0)
#______ 
#______ def pixelcount(regionmask):
#______     return np.sum(regionmask)
#______ 
#______ def rel_size(regionmask):
#______     shp = im.shape
#______     return np.sum(regionmask) /  (shp[0] * shp[1])
#______ 
#______ def extrema(coords):
#______     mins = np.min(coords, axis = 0)
#______     maxs = np.max(coords, axis = 0)
#______     return dict(min_x = mins[1], max_x = maxs[1], min_y = mins[0], max_y = maxs[0])
#______ 
#______ 
#______ props = regionprops(labels, extra_properties=(pixelcount, rel_size,))
#______ 
#______ i_area_lst = list(map(lambda x: [x[0], -x[1].area], enumerate(props)))
#______ area_sorted = sorted(i_area_lst, key = lambda x: x[1])
#______ 
#______ def region_from_top_to_bottom(roi_prop):
#______     """
#______         fn checks if region spans across whole y range
#______         roi_prop: a region created by regionprops fn
#______     """
#______     print("region_from_top_to_bottom___st")
#______     ext = extrema(roi_prop.coords)
#______     print("region_from_top_to_bottom___end")
#______     return ext["min_y"] == 0 and ext["max_y"] == im.shape[0] - 1
#______ 
#______ area_sorted_clup = list(filter(lambda x: not region_from_top_to_bottom(props[x[0]]), area_sorted))
#______ 
#______ flipped = len(area_sorted_clup) < len(area_sorted)
#______ 
#______ region_prop = props[area_sorted_clup[0][0]]
#______ 
#______ width = 150
#______ 
#______ coords = region_prop.coords
#______ extr_val = extrema(coords)
#______ start = extr_val["min_x"]
#______ end = extr_val["max_x"]
#______ center = int((start + end) / 2)
#______ width_raw = end - start
#______ width_report = width_raw if width <= 0 else width
#______ half = int( width_report / 2 )
#______ start = center - half if center - half >= 0 else 0
#______ #    return {"start": start, "width": width_report, "center": center}
#______ print({"start": start, "width": width_report, "center": center})
#______ 
#______ 
#______ 
#______ 
#______ 
#______ for i,p in enumerate(props):
#______     print(f' ------------------------- {i}')
#______     print(f' area: {p.area}')
#______     print(f' coords: {p.coords}')
#______     print(f' pixelcount: {p.pixelcount}')
#______     print(f' rel_size: {p.rel_size}')
#______ 
#______     coord = p.coords
#______     ext = extrema(coord)
#______     print(f' min_x: {ext["min_x"]}')
#______     print(f' min_y: {ext["min_y"]}')
#______     print(f' max_x: {ext["max_x"]}')
#______     print(f' max_y: {ext["max_y"]}')
#______    
#______ 
#______    #print(f' rel_size_region: {p.rel_size_region}')
#______ 
#______ 
#______ 



#_how to run__| cd /D C:\Users\Aigar\miniconda3\
#_how to run__| set LOC="D:\temp_data"
#_how to run__| set AN_SRC="D:\projects\HLD_parameter_determination\hld_ift_analysis_helpers\src\hld_ift_analysis_helpers"
# python %AN_SRC%\test_needle3_bit.py
