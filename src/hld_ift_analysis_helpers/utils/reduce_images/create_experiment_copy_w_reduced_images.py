# script to make a copy of experiment with lower size
#
# inputs: current location, root where it has to be recreated
# 
#  in root create folder with the same experiment name
#   for all subfolders and files that are not of type `scan-???`
#           copy them
#   for all subfolders of type `scan_???`
#       create subfolder in new experiment
#       for all subfolders of type conc_???????
#           create subfolder in new experiment
#           for each file
#           open, find needle image 250 px wide
#           save it in new folder


import os
import re
import shutil
import sys

sys.path.append("D:/projects/HLD_parameter_determination/hld_ift_analysis_helpers/src")

from hld_ift_analysis_helpers.montage_bits import load_image, find_needle_pos
from hld_ift_analysis_helpers.collect_files_folders import collect_data_jsons

import skimage as ski
from skimage.io import imread,imsave

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("source", help = "source of data.json file(s); can be a path to file, a path to file containing list of pathes or a folder")
parser.add_argument("target", help = "target for copy of an experiment(s); this should be a folder ")
#parser.add_argument("-e", "--extraction_options", help = "a path to file that specifies extraction options: default value, value path in a data.json file, target variable name", default = "")
parser.add_argument("-w", "--width", help = "width of image to include, default: 250 px", type = int, default = 250)
args = parser.parse_args()

# select source files
file_path = []
#extraction_options = args.extraction_options if os.path.isfile(args.extraction_options) else ""

def process_string_pointing_to_data_json_file(astr):
    if os.path.split(astr)[1] == "data.json":
        file_path.append(astr)

if os.path.isfile(args.source):
    if os.path.split(args.source)[1] == "data.json":
        # a single source file
        process_string_pointing_to_data_json_file(args.source) 
    else:
        # a file containing list of source files
        with open(args.source, 'r') as file:
            for line in file:
                process_string_pointing_to_data_json_file(line) 

if os.path.isdir(args.source):
    for f in collect_data_jsons(args.source):
        process_string_pointing_to_data_json_file(f)

if len(file_path) == 0:
    print(f'not sure what to do with given source:\n `{args.source}`\n exiting')
    exit()

if not os.path.isdir(args.target):
    print(f'target should be a folder where to create copies of selected experiment.\n   Instead: `{args.target}`\n    was provided.')
    exit()

def experiment_nm(path_data_json):
    return os.path.split(os.path.split(path_data_json)[0])[1]

def copy_experiment_helper(path_data_json, trg_fldr):
    exp_nm = experiment_nm(path_data_json)

    new_trg = os.path.join(trg_fldr, exp_nm)

    if os.path.exists(new_trg):
        print(f'a target folder for experiment:\n   {path_data_json}\nalready exists!!! Will skip it for now. Delete folder:\n   {new_trg}\nand try again.')
        return

    exp_src = os.path.split(path_data_json)[0]

    copy_experiment_w_samller_images(exp_src, trg_fldr)
    

def copy_experiment_w_samller_images(location_in, root_out):
    # do something
    if not contains_scan_folders(location_in):
        print(f'location:\n   {location_in}\nis not an experiment, does not contain scan folders!!!\n ... exiting ...')
        return

    print('---')
    print(f'experiments folder: {os.path.split(location_in)[1]}')
    print(f' root out: {root_out}')
    experiment_copy_target = os.path.join(root_out, os.path.split(location_in)[1])
    print(f'a copy of the experiment will be created at:\n   {experiment_copy_target}')
    os.makedirs(experiment_copy_target, exist_ok = True)

    copy_folder_files(location_in, experiment_copy_target)

    a_scan_folders = scan_folders_in(location_in)
    all_folders = list_folders_in(location_in)
    first = True#DEBUG
    for p in all_folders:
        trg = os.path.join(experiment_copy_target, os.path.split(p)[1])
        if p in a_scan_folders:
            if first: #DEBUG
                first = True #False #DEBUG
                #print(f'skipping a folder:\n   {p}\n for now!')
                print(f'processing scan folder:\n   {p}')
                scan_folder__copy_reduced_images(p, trg) 
            else:
                print(f'skipping a folder:\n   {p}\n for now!')
        else:
            os.makedirs(trg, exist_ok = True)
            copy_folder_recursively(p, trg)

def scan_folder__copy_reduced_images(folder_in, folder_out):
    copy_folder_files(folder_in, folder_out)
    all_folders = list_folders_in(folder_in)
    folders_to_process = list(filter(lambda x: re.search("conc_[0-9.]{7}$", x) is not None, all_folders))

    os.makedirs(folder_out, exist_ok = True)
    print(f' will process following folders: ') 
    for f in folders_to_process:
        print(f'   {f}')
    print()

    for f in folders_to_process:
        print(f'processing folder:\n   {f}')
        measurement_folder__copy_reduced_images(f, os.path.join(folder_out, os.path.split(f)[1]))

def measurement_folder__copy_reduced_images(folder_in, folder_out):
    raw = list(map(lambda x: os.path.join(folder_in, x), os.listdir(folder_in)))
    images_in = list(filter(lambda x: not os.path.isdir(x) and re.search(".jpg$", x) is not None, raw))
    images_out = list(map(lambda x: os.path.join(folder_out, os.path.split(x)[1]), images_in)) 

    os.makedirs(folder_out, exist_ok = True)

    needle_pos = find_needle_pos(images_in[0], args.width)
    start = needle_pos["start"]
    width = needle_pos["width"]
    print(f'start: {start}, width: {width}')
    N_to_process = -1
    for index, im_in, im_out in zip(range(len(images_in)), images_in, images_out):
        if index < N_to_process or N_to_process < 0:
            print(f'{index}: {im_in} --> {im_out}')
            #>>> #im = load_image(im_in, start, width)
            im = imread(im_in)
            #>>> print(f'im.shape: {im.shape}; type: {im.dtype}')
            im = im[:, start:start+width, :]
            imsave(im_out, im)
            #>>> im2 = imread(im_out)

            #>>> print(im)
            #>>> print("----")
            #>>> #print(ski.util.img_as_ubyte(im))
            #>>> #print("----")
            #>>> print(im2)
            #>>> print()


def scan_folders_in(folder_in):
    dirs = list_folders_in(folder_in)
    return list(filter(lambda x: re.search("scan_[0-9]{3}$", x) is not None, dirs))

def contains_scan_folders(folder_in):
    return len(scan_folders_in(folder_in)) > 0

def list_folders_in(folder_in):
    raw = list(map(lambda x: os.path.join(folder_in, x), os.listdir(folder_in)))
    return list(filter(lambda x: os.path.isdir(x), raw))

def copy_folder_files(folder_in, folder_out):
    raw = list(map(lambda x: os.path.join(folder_in, x), os.listdir(folder_in)))
    files = list(filter(lambda x: not os.path.isdir(x), raw))
    
    print(f'copying content of: \n   {folder_in}\n ... to ...\n   {folder_out}')
    for f in files:
        target = os.path.join(folder_out, os.path.split(f)[1])
        print(f'{f} --> {target}')
        dest = shutil.copyfile(f, target)

def copy_folder_recursively(folder_in, folder_out):
    copy_folder_files(folder_in, folder_out)
    folders = list_folders_in(folder_in)
    for p in folders:
        trg = os.path.join(folder_out, os.path.split(p)[1])
        print(f'creating folder:\n   {trg}')
        os.makedirs(trg, exist_ok = True)
        copy_folder_recursively(p, trg)


for i,p in enumerate(file_path):
    print(f'*******************************************************************************')
    print(f'                                                                 {i} of {len(file_path)}')
    print(f'   from:')
    print(f'      {p}')
    print(f'   to:')
    print(f'      {args.target}')
    print(f'*******************************************************************************')
    copy_experiment_helper(p, args.target)

#___-### test that should not work
#___-#src = r'\\huckdfs-srv.science.ru.nl\huckdfs\RobotLab\Storage-Miscellaneous\aigars\temp\scripts'
#___-#trg = r'\\huckdfs-srv.science.ru.nl\huckdfs\RobotLab\Storage-Miscellaneous\aigars\temp\exp_reduced'
#___-#copy_experiment_w_samller_images(src, trg)
#___-
#___-
#___-src = [
#___-        r'\\huckdfs-srv.science.ru.nl\huckdfs\RobotLab\Storage-Miscellaneous\aigars\temp\BrijL4_C7C16_NaCl_experiments_raw\exp_2025-03-26_05g_BrijL4_C7C16_001',
#___-        r'\\huckdfs-srv.science.ru.nl\huckdfs\RobotLab\Storage-Miscellaneous\aigars\temp\BrijL4_C7C16_NaCl_experiments_raw\exp_2025-03-26_06g_BrijL4_C7C16_001',
#___-        r'\\huckdfs-srv.science.ru.nl\huckdfs\RobotLab\Storage-Miscellaneous\aigars\temp\BrijL4_C7C16_NaCl_experiments_raw\exp_2025-03-27_15g_BrijL4_C7C16_001'
#___-        ]
#___-trg = r'\\huckdfs-srv.science.ru.nl\huckdfs\RobotLab\Storage-Miscellaneous\aigars\temp\exp_reduced'
#___-#copy_experiment_w_samller_images(src[0], trg)
#___-#copy_experiment_w_samller_images(src[1], trg)
#___-#copy_experiment_w_samller_images(src[2], trg)
#___-
#___-#src = r'\\huckdfs-srv.science.ru.nl\huckdfs\RobotLab\Storage-Miscellaneous\aigars\temp\exp_2025-03-25_04g_BrijL4_C7C16_001'
#___-#trg = r'\\huckdfs-srv.science.ru.nl\huckdfs\RobotLab\Storage-Miscellaneous\aigars\temp\exp_reduced'
#___-#copy_experiment_w_samller_images(src, trg)
#___-
#___-#src = r"\\huckdfs-srv.science.ru.nl\huckdfs\RobotLab\Storage-Miscellaneous\aigars\temp\end_config.json"
#___-#trg = r"D:\tenp_data_2\end_config.json"
#___-#shutil.copyfile(src,trg)


