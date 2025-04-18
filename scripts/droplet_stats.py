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
from skimage.util import img_as_ubyte
from skimage.draw import ellipse
from skimage.measure import label, regionprops, regionprops_table
from skimage.transform import rotate
from skimage.filters import rank
from skimage.segmentation import watershed, mark_boundaries
from skimage import feature
from skimage.morphology import (
                                reconstruction,
                                erosion, 
                                binary_erosion,
                                binary_closing,
                                binary_dilation,
                                disk
                                )

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

def process_images(im_pathes, csv_path, fn, test = 5):
    """
    fn finds droplets in images within given folder, see `process_image()` 
    """
    
    for i, p in zip(range(len(im_pathes)), im_pathes):
        if i < test or test < 0:
            df = fn(p, i)
            if os.path.exists(csv_path):
                df.to_csv(csv_path, mode='a', index=False, header=False)
            else:
                df.to_csv(csv_path, mode='w', index=False, header=True)
        else: 
            return

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


#def droplet_distribution
#   min, max, stdev along x
#   min, max, stdev along y
#                               distributions are rescaled with respect to needle width
#               

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

# ================================================================================
#                                                           needle roi 
def needle(im, width = 0):
    """
    fn finds x coordinates range for needle in given image

    fn expects a uniform image with a low intensity in a narrow x coordinate range
    """
    im_gray = im if len(im.shape) == 2 else rgb2gray(im) 
    im_std = im_gray.std(axis = 0)
    im_max = im_std.max()
    im_min = im_std.min()
    #print(im_std)
    #print(im_max)
    #print(im_min)

    above_threshold =  list(filter(lambda x: x > (im_max + im_min)/2, im_std.tolist()))
    index = np.where(im_std > (im_max + im_min)/2)[0]
    #print(f'index above average: {index}')
    center = int(sum(index) / len(index))
    #start = min(index)
    start = int(index[0])
    #width = max(index) - min(index)
    width_raw = int(index[len(index) - 1] - start)
    width_report = width_raw if width == 0 else width
    half = int( width_report / 2 )
    start = center - half if center - half >= 0 else 0
    return {"start": start, "width": width_report, "center": center}


def boolean_vector_to_regions(vector):
    return list(filter(lambda x: x[1] == 1, 
                        list(map(lambda x: (x[0], 1 if x[1] else 0),
                                zip(range(len(vector)), vector)))))

def join_adjacent_regions(lst_regions):
    def __helper_fn(lst, region):
        if lst == []:
            return [region]
        last = lst[len(lst) - 1]
        if last[0] + last[1] == region[0]:
            new_last = (last[0], last[1] + 1)
            lst[len(lst) - 1] = new_last
        else:
            lst.append(region)
        return lst
    return list(functools.reduce(__helper_fn, lst_regions, []))

## test bits ------------------------
#x = [ a > 10 or a < 5 for a in range(30) ]
#y = boolean_vector_to_regions(x)
#z = join_adjacent_regions(y)
#print(x)
#print(y)
#print(z)
## end test bits =------------------------


# =======================================================
# general notes on dripping characterization
#
#   find needle region
#   find boundary between needle and area below
#   segment area below needle
#   
# need to setup workflow to
#   open an image(s)
#    find needle, region below needle, show segmentation results


#def needle_roi(im, width) -> ndarray:
#

def load_image(a_path):
    return imread(a_path, as_gray = True)

def needle_image(im, width = 0):
    needle_params = needle(im, width)
    st = needle_params["start"]
    w =  needle_params["width"]
    img_shape = len(im.shape)
    if img_shape == 2:
        return im[:, st:st+w]
    elif img_shape == 3:
        return im[:, st:st+w, :]
    else:
        return None

def region_below_needle(im, max_weight = 0.5) -> list:
    """
    fn finds y coordinate where needle has lowest point

    
    """
    #print(f'im.shape: {im.shape}')#debug
    im_std = im.std(axis = 1)
    im_std_max = im_std.max()
    im_std_min = im_std.min()

    #print(f'im_std values: {im_std}') #debug
    #print(f'im_std values: {im_std[600:700]}') #debug
    #print(f'im_std_min: {im_std_min}\nim_std_max: {im_std_max}')

    #max_weight = 0.75
    threshold = max_weight * im_std_max + (1 - max_weight) * im_std_min
    #print(f'max_weight: {max_weight}\nthreshold: {threshold}\nim_std.shape: {im_std.shape}')#debug 

    #above_threshold =  list(filter(lambda x: x < threshold, im_std.tolist()))
    #print(f'above_threshold: {above_threshold}')

    index = np.where(im_std < threshold)[0]
    print(f'region_below_needle(): index: {index}')
    

    #im_min = im.min(axis = 1)
    #im_min_max = im_min.max()
    #im_min_min = im_min.min()

    #print(f'im_min values: {im_min}') #debug

    #above_threshold =  list(filter(lambda x: x < (im_min_max + im_min_min)/2, im_min.tolist()))
    #print(f'above_threshold: {above_threshold}')

    #index = np.where(im_min < (im_min_max + im_min_min)/2)[0]
    #print(f'index: {index}')

    #> #print(f'index above average: {index}')
    #> center = int(sum(index) / len(index))
    #> #start = min(index)
    #> start = int(index[0])
    #> #width = max(index) - min(index)
    #> width_raw = int(index[len(index) - 1] - start)
    #> width_report = width_raw if width == 0 else width
    #> half = int( width_report / 2 )
    #> start = center - half if center - half >= 0 else 0
    #> return {"start": start, "width": width_report, "center": center}
    return im[index.min():, :]

#def below_needle_roi(im) -> ndarray:
#

def get_edges_image(im):
    return feature.canny(im, sigma = 3)

def dripping_objects(im):
    image = img_as_ubyte(im)

    # denoise image
    denoised = rank.median(image, disk(2))
    
    #find regions with high gradient values (outer boundaries of drops, streams)
    markers = rank.gradient(denoised, disk(1)) > 10 # original disk(5), < 10

    #fill in gaps and reduces a bit in size
    #> #__v1__
    #> seed = np.copy(markers)
    #> seed[1:-1, 1:-1] = image.max()
    #> mask = markers
    #> filled = reconstruction(seed, mask, method='erosion')    

    #> filled = erosion(filled, disk(2))

    #__v2__
    temp = binary_dilation(markers, footprint = disk(1))
    temp = binary_closing(temp, footprint = disk(5))
    filled = binary_erosion(temp, footprint = disk(4))


    #labeled regions reported
    return ndi.label(filled)[0]
    

def load_region_below_needle(path, width = 0, max_weight = 0.8):
    im = load_image(path)
    needle_reg_im = needle_image(im, width)
    return region_below_needle(needle_reg_im, max_weight)

def file_path(root, id, label):
    return os.path.join(root, f'{id:07d}_{label}.jpg')

def process_dripping_stats(path, root, id, width = 0, max_weight = 0.8, save_raw_region = True, save_object_image = True):
    original_region_path = file_path(root, id, "original")
    object_region_path = file_path(root, id, "object_regions")
    im = load_region_below_needle(path, width, max_weight)
    dripping_im = dripping_objects(im)

    props = regionprops_table(
        dripping_im,
        properties=(
                'label',
                'area',
                'eccentricity',
                'bbox',
                'centroid',
                'orientation',
                'axis_major_length',
                'axis_minor_length'
                ),
    )

    df = pd.DataFrame(props)
    df['r'] = (df['bbox-2'] - df['bbox-0'] + df['bbox-3'] - df['bbox-1'] ) / 4
    #df['area_exp'] = df['r'] * df['r'] * pi 
    #df['q'] = abs(df['area'] / df['area_exp'] - 1)
    #df = df[(df['q'] < 0.05) & (df['r'] > 2)]
    df['path'] = path
    df['id'] = id
    df['original_region_path'] = original_region_path 
    df['object_region_path'] = object_region_path 


    imsave(original_region_path, img_as_ubyte(im))
    imsave(object_region_path, img_as_ubyte(dripping_im))

    return df

