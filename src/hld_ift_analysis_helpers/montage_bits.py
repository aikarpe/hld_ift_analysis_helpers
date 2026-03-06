#================================================================================
#   functions to create montages from HLD_IFT experimental data
#================================================================================
#import sys 
#sys.path.append("D:/projects/HLD_parameter_determination/hld_ift_analysis_helpers/scripts")

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
from hld_ift_analysis_helpers.collect_files_folders import collect_images
from hld_ift_analysis_helpers.locations import montage_name__experiment
from functools import reduce
# ================================================================================
#                                                           needle roi 

#> def needle(im, width = 0):
#>     """
#>     fn finds x coordinates range for needle in given image
#> 
#>     fn expects a uniform image with a low intensity in a narrow x coordinate range
#>     """
#>     im_std = im.std(axis = 0)
#>     im_max = im_std.max()
#>     im_min = im_std.min()
#> 
#>     above_threshold =  list(filter(lambda x: x > (im_max + im_min)/2, im_std.tolist()))
#>     index = np.where(im_std > (im_max + im_min)/2)[0]
#>     #print(f'index above average: {index}')
#>     center = int(sum(index) / len(index))
#>     #start = min(index)
#>     start = int(index[0])
#>     #width = max(index) - min(index)
#>     width_raw = int(index[len(index) - 1] - start)
#>     width_report = width_raw if width == 0 else width
#>     half = int( width_report / 2 )
#>     start = center - half if center - half >= 0 else 0
#>     return {"start": start, "width": width_report, "center": center}

def needle(im, width = 0):
    def pixelcount(regionmask):
        return np.sum(regionmask)
    
    def rel_size(regionmask):
        shp = im.shape
        return np.sum(regionmask) /  (shp[0] * shp[1])
    
    def extrema(coords):
        mins = np.min(coords, axis = 0)
        maxs = np.max(coords, axis = 0)
        return dict(min_x = mins[1], max_x = maxs[1], min_y = mins[0], max_y = maxs[0])
    
    def region_from_top_to_bottom(roi_prop):
        """
            fn checks if region spans across whole y range
            roi_prop: a region created by regionprops fn
        """
        #print("region_from_top_to_bottom___st")
        ext = extrema(roi_prop.coords)
        #print("region_from_top_to_bottom___end")
        return ext["min_y"] == 0 and ext["max_y"] == im.shape[0] - 1
    
    def region_at_image_side_edges(roi_prop):
        """
            fn checks if region spans across whole y range
            roi_prop: a region created by regionprops fn
        """
        #print("region_from_top_to_bottom___st")
        ext = extrema(roi_prop.coords)
        #print("region_from_top_to_bottom___end")
        return ext["min_x"] == 0 or ext["max_x"] == im.shape[1] - 1
    
    im_max = im.max()
    #print(f'im_max: {im_max}')
    
    im2 = im_max - im
        
    threshold = ski.filters.threshold_otsu(im2)
    
    mask = im2 > threshold
    
    mask = ski.morphology.remove_small_objects(mask, 50)
    mask = ski.morphology.remove_small_holes(mask, 50)
    
    labels = ski.measure.label(mask)
    
    props = ski.measure.regionprops(labels, extra_properties=(pixelcount, rel_size,))
    
    i_area_lst = list(map(lambda x: [x[0], -x[1].area], enumerate(props)))
    i_area_lst_sorted = sorted(i_area_lst, key = lambda x: x[1])
    
    i_area_lst_cleaned_up = list(filter(lambda x: not region_from_top_to_bottom(props[x[0]]), i_area_lst_sorted))
    i_area_lst_cleaned_up = list(filter(lambda x: not region_at_image_side_edges(props[x[0]]), i_area_lst_cleaned_up))
    
    flipped = len(i_area_lst_cleaned_up) < len(i_area_lst_sorted)
    
    region_prop = props[i_area_lst_cleaned_up[0][0]]
    
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
    return {"start": start, "width": width_report, "center": center}

    # threshold image to binary
    #   remove small holes and small specks
    # label image
    # for all regions get all pixels involved
    # discard regions if:
    #       y spans whole region of image
    #       x(max) - x(min) is above width_min and below width_max
    #       

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

def load_image(a_path, x_start, width, create_empty = False, height = 1500, default_value = 0):
    try:
        im = imread(a_path, as_gray = True)
        return im[: , x_start:x_start + width]
    except:
        return np.full([height, width], 0, dtype = np.uint8)


def montage_row(lst_ims, padding_width = 0, fill = 0):
    return ski.util.montage(lst_ims, grid_shape = (1, len(lst_ims)), padding_width = padding_width, fill = fill)

def montage_col(lst_ims, padding_width = 0, fill = 0):
    return ski.util.montage(lst_ims, grid_shape = (len(lst_ims), 1), padding_width = padding_width, fill = fill)

def montage_row_from_path(lst_path, x_start, width, padding_width = 0, fill = 0):
    return montage_row(list(map(lambda x: ski.img_as_ubyte(load_image(x, x_start, width)), lst_path)), padding_width = padding_width, fill = fill)

def make_lst_im_same_shape(lst_ims): 
    def shape_helper(mins, im):
        shp = im.shape
        return (min(mins[0], shp[0]), min(mins[1], shp[1]))

    shape_common = reduce(shape_helper, lst_ims, (9999999999,9999999999))
   
    all_same = reduce(lambda status, image: status and image.shape[0] == shape_common[0] and image.shape[1] == shape_common[1], lst_ims, True)
    if not all_same:
        print(f'not all images are the same in input list! common shape is: {shape_common}')
    return lst_ims if all_same else list(map(lambda x: x[0:shape_common[0], 0:shape_common[1]], lst_ims))
# ================================================================================
#                                                           dataframe manipulation

def aggregate_item(df, group_by, aggregate_by, fn):
    output = df.groupby(by = group_by)[aggregate_by].agg(fn) if type(aggregate_by) == list else df.groupby(by = group_by)[[aggregate_by]].agg(fn) 
    print(f'aggregate_item(): debug: index: {output.index}')
    print(f'aggregate_item(): debug: index: {output.index.names}')
    df_new = output.index.to_frame()
    vars_to_process = aggregate_by if type(aggregate_by) == list else [aggregate_by]
    for var in vars_to_process:
        df_new[var] = output[var].tolist()
    return df_new.reset_index(drop = True)

#>>>> # Source - https://stackoverflow.com/a
#>>>> # Posted by Scott Boston, modified by community. See post 'Timeline' for change history
#>>>> # Retrieved 2026-01-16, License - CC BY-SA 3.0
#>>>> 
#>>>> df.groupby('Category').agg({'Item':'size','shop1':['sum','mean','std'],'shop2':['sum','mean','std'],'shop3':['sum','mean','std']})

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
    if os.path.exists(path):
        return needle(imread(path, as_gray = True), width = width)
    else:
        return {"start": 0}


# ===============================================
# montages to make

def conditional_reverse(status):
    """ 
        fn reverses list if status == TRUE, otherwise 
            list remains in the same order
    """
    if status:
        return lambda x: x[::-1]
    else:
        return lambda x: x[::1]


def make_montage_of_measurement(root, i_start = 1, n_images = 5, roi_width = 200, test = -1, output_folder = ""):
    use_default_fldr = output_folder == ""
    im_data = collect_images_to_dataframe(root)
    
    def select_start_to_end(i_start): 
        return lambda x: list(x.tolist())[i_start:]
    def select_N_from_start(i_start, n):
        return lambda x: list(x.tolist())[i_start:i_start + n]
    fn = select_start_to_end(i_start) if n_images < 1 else select_N_from_start(i_start, n_images)
    temp = aggregate_item(im_data, ['experiment_root', 'experiment', 'scan', 'concentration'], "path", fn)
    temp['roi_x_start'] = list(map(lambda x: find_needle_pos(x[0], width = roi_width)["start"], temp['path']))
    temp['roi_x_width'] = roi_width
    temp['montage_name'] = list(map(lambda x: scan(x[0]) + "_" + concentration_bit(x[0]) + "_measurement_montage.png", 
                                    temp['path'], ))
    temp['montage_folder'] = list(map(lambda x: os.path.join(experiment_root(x[0]), "meas_montage") if use_default_fldr else output_folder, 
                                    temp['path'], ))

    temp['montage_path'] = list(map(lambda x,y: os.path.join(x,y), 
                                    temp['montage_folder'],
                                    temp['montage_name']))

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
            os.makedirs(os.path.split(p)[0], exist_ok = True)
            imsave(p, ski.util.img_as_ubyte(im))
    print(f'... done making montage in `{root}`')
   
    

def make_montage_of_experiment_df(im_data, i_start = 1, n_images = 5, roi_width = 200, test = 5, output_path = "", reverse_measurement_order = False, reverse_scan_order = False, transpose_scan_measurement = False):
    """
    im_data [pandas DataFrame] with following columns
        - path: String, path to image
        - experiment: String, experiment name (also folder name)
        - experiment_root: String, path to experiment root
        - scan: String, scan name of type "scan_<000>" where <000> is scan index
        - concentration: String, measurement label of type "conc_<x.xxxxx>"
        - concentration_val: float, relative concentration of a scan (float of "<x.xxxxx>")
        - image_name: String: file name of type "<XXXXX>.jpg", were XXXXX is index
        - image_index: int, image index in measurement
    """
    #> changes to add!!!!
    #>  - reverser for montage_col and montage_row
    #>  - flip scan and conc coordinates
     
    order_measurements = conditional_reverse(reverse_measurement_order)
    order_scans = conditional_reverse(reverse_scan_order)

    var2            = 'concentration'    if transpose_scan_measurement else 'scan'
    order_first     = order_scans        if transpose_scan_measurement else order_measurements
    order_second    = order_measurements if transpose_scan_measurement else order_scans

    #temp = aggregate_item(im_data, ['experiment_root', 'experiment', 'scan', 'concentration'], "path", lambda x: list(x.tolist())[i_start:i_start+n_images])
    temp = aggregate_item(im_data, ['experiment', 'scan', 'concentration'], "path", lambda x: list(x.tolist())[i_start:i_start+n_images])
    temp['roi_x_start'] = list(map(lambda x: find_needle_pos(x[0], width = roi_width)["start"], temp['path']))
    temp['roi_x_width'] = roi_width
    temp['montage'] = list(map(lambda x,y,z: montage_row_from_path(x, y, z, padding_width = 0), temp['path'], temp['roi_x_start'], temp['roi_x_width']))
    def montage_log_make(alst, sep):
        out = ""
        for x in alst:
            out += x + sep
        return out
    temp['montage_log'] = list(map(lambda x: montage_log_make(x,"\n"), temp['path']))
    
    #>>>
    print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
    print(temp['path'].tolist())
    print("--------------------------------------------------------------------------------")
    print(list(map(lambda x: x.shape, temp['montage'])))
    print("--------------------------------------------------------------------------------")
    print(temp['montage_log'].tolist())
    print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
    #>>>

    
    # temp = aggregate_item(temp, ['experiment_root', 'experiment', 'scan'], 'montage', lambda x: montage_row(x.tolist(), padding_width = 5))
    temp = aggregate_item(temp, ['experiment', var2], ['montage','montage_log'], lambda x: x.to_list())
    temp['montage'] = list(map(lambda x: montage_row(order_first(make_lst_im_same_shape(x)), padding_width = 5), temp['montage']))
    temp['montage_log'] = list(map(lambda x: montage_log_make(order_first(x),"\n---COL_SEP---\n\n\n"), temp['montage_log']))
    #>>>
    print("2$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
    print(list(map(lambda x: x.shape, temp['montage'])))
    print(temp['montage_log'].tolist())
    print("2$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
    #>>>


    #temp = aggregate_item(temp, ['experiment_root', 'experiment'], 'montage', lambda x: montage_col(x.tolist(), padding_width = 10))
    temp = aggregate_item(temp, ['experiment'], ['montage','montage_log'], lambda x: x.to_list())
    temp['montage'] = list(map(lambda x: montage_col(order_second(make_lst_im_same_shape(x)), padding_width = 10), temp['montage']))
    temp['montage_log'] = list(map(lambda x: montage_log_make(order_second(x),"\n======ROW=SEP======\n\n\n\n\n"), temp['montage_log']))
    temp['montage_path'] = list(map(
                                lambda t_exp: montage_name__experiment(t_exp, i_start, n_images, roi_width), 
                                temp['experiment']
                                ))

    #>>>
    print("3$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
    print(list(map(lambda x: x.shape, temp['montage'])))
    print(temp['montage_log'].tolist())
    print("3$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
    #>>>


    for p,im in zip(temp['montage_path'], temp['montage']):
        montage_path = p if output_path == "" else output_path
        print(type(im))
        print(f'... saving {montage_path}')
        imsave(montage_path, ski.util.img_as_ubyte(im))

def make_montage_of_experiment(root, i_start = 1, n_images = 5, roi_width = 200, test = 5, output_path = "", reverse_measurement_order = False, reverse_scan_order = False, transpose_scan_measurement = False):
    im_data = collect_images_to_dataframe(root)
    make_montage_of_experiment_df(im_data, i_start = i_start, n_images = n_images , roi_width = roi_width, test = test, output_path = output_path, reverse_measurement_order = reverse_measurement_order, reverse_scan_order = reverse_scan_order, transpose_scan_measurement = transpose_scan_measurement)
    print(f'... done making montage in `{root}`')

def make_montage_of_experiment_csv(csv_path, i_start = 1, n_images = 5, roi_width = 200, test = 5, output_path = "", reverse_measurement_order = False, reverse_scan_order = False, transpose_scan_measurement = False): 
    im_data = pd.read_csv(csv_path)
    make_montage_of_experiment_df(im_data, i_start = i_start, n_images = n_images , roi_width = roi_width, test = test, output_path = output_path, reverse_measurement_order = reverse_measurement_order, reverse_scan_order = reverse_scan_order, transpose_scan_measurement = transpose_scan_measurement)
    print(f'... done making montage from `{csv_path}`')

