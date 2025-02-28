#================================================================================
#   script takes input image and creates list of blobs seen in it
#================================================================================
import sys 
sys.path.append("D:/projects/HLD_parameter_determination/hld_ift_analysis_helpers/scripts")

import os
import re

import numpy as np
import matplotlib.pyplot as plt

import skimage as ski
from skimage.color import rgb2gray
from skimage.io import imread,imsave

from scipy import ndimage as ndi
from skimage.draw import ellipse
from skimage.measure import label, regionprops, regionprops_table
from skimage.transform import rotate
import pandas as pd
from math import pi
from collect_files_folders import list_images

# ================================================================================
#                                                           find droplets and report
#                                                           their stats

def process_image(path):
    """
    fn finds droplets in an image and reports their parameters as dataframe
    """
    print(f'processing:]n{path}')

    image = imread(path, as_gray=False)
    image_gray = rgb2gray(image)

    coins = image_gray #[: , 800:950]
    edges = ski.feature.canny(coins)

    # ============================================================ canny edge detection
    #                                                               and filling found edges

    fill_coins = ndi.binary_fill_holes(edges)
    # ============================================================ label segments

    labeled_coins, _ = ndi.label(fill_coins)
    image_label_overlay = ski.color.label2rgb(labeled_coins, image=coins, bg_label=0)
    # ============================================================ report stats on labeled segments

    props = regionprops_table(
        labeled_coins,
        properties=('label', 'area', 'eccentricity', 'bbox', 'centroid'),
    )
    df = pd.DataFrame(props)
    df['r'] = (df['bbox-2'] - df['bbox-0'] + df['bbox-3'] - df['bbox-1'] ) / 4
    df['area_exp'] = df['r'] * df['r'] * pi 
    df['q'] = abs(df['area'] / df['area_exp'] - 1)
    df = df[(df['q'] < 0.05) & (df['r'] > 2)]
    df['path'] = path
    return df

def process_dir(path, csv_path):
    """
    fn finds droplets in images within given folder, see `process_image()` 
    """
    im_pathes = list_images(path)
    for i, p in zip(range(len(im_pathes)), im_pathes):
        #if i > 25:
        #    return
        df = process_image(p)
        if os.path.exists(csv_path):
            df.to_csv(csv_path, mode='a', index=False, header=False)
        else:
            df.to_csv(csv_path, mode='w', index=False, header=True)


#================================================================================
#> # this part generates all output from all available droplets!!! takes a very long time
#> folders_use = match_conc_at_end(collect_dirs(raw_data_folder))
#> #folders_use = [folders_use[0]]
#> 
#> first = True
#> path_out = os.path.join(raw_data_folder, "data.output.csv")
#> if os.path.exists(path_out):
#>     os.remove(path_out)
#> for p in folders_use:
#>     print(p)
#>     process_dir(p, path_out)
#>     
#================================================================================


