#================================================================================
#   functions to create montages from HLD_IFT experimental data
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
from collect_files_folders import collect_images
# ================================================================================
#                                                           needle roi 
def needle(im, width = 0):
    """
    fn finds x coordinates range for needle in given image

    fn expects a uniform image with a low intensity in a narrow x coordinate range
    """
    im_std = im.std(axis = 0)
    im_max = im_std.max()
    im_min = im_std.min()

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

# ================================================================================
#                                                            path manipulation
#                                                            to extract process variables 
#                                                            of HLD_IFT scan 

def path_split(path, n = 1):
    out = os.path.split(path)
    if n < 2:
        return out
    else:
        return path_split(out[0], n = n - 1)
def experiment(path):
    return path_split(path, n = 4)[1]
def experiment_root(path):
    return path_split(path, n = 3)[0]
def scan(path):
    return path_split(path, n = 3)[1]
def scan_root(path):
    return path_split(path, n = 2)[0]
def concentration_bit(path):
    return path_split(path, n = 2)[1]
def concentration(path):
    return re.sub("conc_", "", concentration_bit(path))
def concentration_root(path):
    return path_split(path, n = 1)[0]
def concentration_val(path):
    return float(concentration(path))
def image_name(path):
    return path_split(path, n = 1)[1]
def image_index(path):
    return int(re.sub("\\..*", "", image_name(path)))


# ================================================================================
#                                                            montage mechanics

def load_image(a_path, x_start, width):
    im = imread(a_path, as_gray = True)
    return im[: , x_start:x_start + width]

def montage_row(lst_ims, padding_width = 0, fill = 0):
    return ski.util.montage(lst_ims, grid_shape = (1, len(lst_ims)), padding_width = padding_width, fill = fill)

def montage_col(lst_ims, padding_width = 0, fill = 0):
    return ski.util.montage(lst_ims, grid_shape = (len(lst_ims), 1), padding_width = padding_width, fill = fill)

def montage_row_from_path(lst_path, x_start, width, padding_width = 0, fill = 0):
    return montage_row(list(map(lambda x: ski.img_as_ubyte(load_image(x, x_start, width)), lst_path)), padding_width = padding_width, fill = fill)


# ================================================================================
#                                                           dataframe manipulation

def aggregate_item(df, group_by, aggregate_by, fn):
    output = df.groupby(by = group_by)[[aggregate_by]].agg(fn)
    print(f'aggregate_item(): debug: index: {output.index}')
    print(f'aggregate_item(): debug: index: {output.index.names}')
    df_new = output.index.to_frame()
    df_new[aggregate_by] = output[aggregate_by].tolist()
    return df_new.reset_index(drop = True)

def reinject_grouped_variables(df):
    index_df = df.index.to_frame()
    for nm in df.index.names:
        df[nm] = index_df[nm]
    return df

def collect_images_to_dataframe(root):
    ims = collect_images(root)

    im_data = pd.DataFrame(data = {"path": ims})
    im_data['experiment']           = list(map(experiment       , im_data['path']))
    im_data['experiment_root']      = list(map(experiment_root  , im_data['path']))
    im_data['scan']                 = list(map(scan             , im_data['path']))
    im_data['concentration']        = list(map(concentration    , im_data['path']))
    im_data['concentration_val']    = list(map(concentration_val, im_data['path']))
    im_data['image_name']           = list(map(image_name       , im_data['path']))
    im_data['image_index']          = list(map(image_index      , im_data['path']))
    return im_data

def find_needle_pos(path, width):
    return needle(imread(path, as_gray = True), width = width)


# ===============================================
# montages to make

def make_montage_of_measurement(root, i_start = 1, n_images = 5, roi_width = 200, test = -1):
    im_data = collect_images_to_dataframe(root)
    
    def select_start_to_end(i_start): 
        return lambda x: list(x.tolist())[i_start:]
    def select_N_from_start(i_start, n):
        return lambda x: list(x.tolist())[i_start:i_start + n]
    fn = select_start_to_end(i_start) if n_images < 1 else select_N_from_start(i_start, n_images)
    temp = aggregate_item(im_data, ['experiment_root', 'experiment', 'scan', 'concentration'], "path", fn)
    temp['roi_x_start'] = list(map(lambda x: find_needle_pos(x[0], width = roi_width)["start"], temp['path']))
    temp['roi_x_width'] = roi_width
    temp['montage_path'] = list(map(lambda x: os.path.join(
                                                experiment_root(x[0]), 
                                                "meas_montage", 
                                                scan(x[0]) + "_" + concentration_bit(x[0]) + "_measurement_montage.png"), 
                                    temp['path'], ))
    #temp['montage'] = list(map(lambda x,y,z: montage_row_from_path(x, y, z, padding_width = 0), temp['path'], temp['roi_x_start'], temp['roi_x_width']))

    for p, pths, start, width, index in zip(temp['montage_path'], 
                                            temp['path'], 
                                            temp['roi_x_start'],
                                            temp['roi_x_width'],
                                            range(len(temp['montage_path']))):
        if test < 0 or index < test: 
            im = montage_row_from_path(pths, start, width)
            print(type(im))
            print(f'... saving {p}')
            imsave(p, ski.util.img_as_ubyte(im))
    print(f'... done making montage in `{root}`')
   
    
def make_montage_of_experiment(root, i_start = 1, n_images = 5, roi_width = 200, test = 5):
    im_data = collect_images_to_dataframe(root)
    
    temp = aggregate_item(im_data, ['experiment_root', 'experiment', 'scan', 'concentration'], "path", lambda x: list(x.tolist())[i_start:i_start+n_images])
    temp['roi_x_start'] = list(map(lambda x: find_needle_pos(x[0], width = roi_width)["start"], temp['path']))
    temp['roi_x_width'] = roi_width
    temp['montage'] = list(map(lambda x,y,z: montage_row_from_path(x, y, z, padding_width = 0), temp['path'], temp['roi_x_start'], temp['roi_x_width']))

    temp = aggregate_item(temp, ['experiment_root', 'experiment', 'scan'], 'montage', lambda x: montage_row(x.tolist(), padding_width = 5))

    temp = aggregate_item(temp, ['experiment_root', 'experiment'], 'montage', lambda x: montage_col(x.tolist(), padding_width = 10))

    temp['montage_path'] = list(map(
                                lambda t_root, t_exp: os.path.join(t_root, f'montage__{t_exp}_i-{i_start}-{n_images}_x-{roi_width}.png'), 
                                temp['experiment_root'],
                                temp['experiment']
                                ))

    for p,im in zip(temp['montage_path'], temp['montage']):
        print(type(im))
        print(f'... saving {p}')
        imsave(p, ski.util.img_as_ubyte(im))
    print(f'... done making montage in `{root}`')



