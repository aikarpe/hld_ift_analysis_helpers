import sys
#> sys.path.append("D:/projects/HLD_parameter_determination/hld_ift_analysis_helpers/src")

from hld_ift_analysis_helpers.montage_bits import *
from hld_ift_analysis_helpers.collect_files_folders import collect_data_jsons
from hld_ift_analysis_helpers.locations import data_json_path_to_measurement_montage_fldr_path

from skimage.io import imread,imsave

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("source", help = "source of data.json file(s); can be a path to file, a path to file containing list of pathes or a folder")
parser.add_argument("-i", "--i_start", help = "first index to use for montage, default: 0", type = int, default = 0)
parser.add_argument("-n", "--n_images", help = "number of images per measurement to include, default: -1, all", type = int, default = -1)
parser.add_argument("-w", "--width", help = "width of image to include, default: 150 px", type = int, default = 150)
parser.add_argument("-t", "--test", help = "images to test, default: -1, all", type = int, default = -1)
parser.add_argument("-o", "--output_folder", help = "output folder where montages are to be saved", type = str, default = "")
parser.add_argument("-p", "--input_string", help = "input string containing comma delimited pathes to all experiments to be processed", type = str, default = "")
parser.add_argument("-x", "--roi_x_start", help = "x coordinate for beginning of needle roi, default: -1, needle roi determined automatically", type = int, default = -1)

args = parser.parse_args()

print(args.source)
print(args.i_start)
print(args.n_images)
print(args.width)
print(args.test)
print(args.output_folder)
print(args.input_string)
print(args.roi_x_start)

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

if args.input_string != "":
    elements = args.input_string.split(",")
    for e in elements:
        if os.path.isfile(e) and os.path.split(e)[1] == "data.json":
            process_string_pointing_to_data_json_file(e)

if len(file_path) == 0:
    print(f'not sure what to do with given source:\n `{args.source}`\n exiting')
    exit()


def find_needle_pos(path, width = 0):
    if os.path.exists(path):
        return needle(imread(path, as_gray = True), width = width)
    else:
        return {"start": -1, "width": -1, "center": -1}

output = []

for fp in file_path:
    
    k = "y"
    #k = input(f'process? (y,n) {fp}')
    if k == "y":
        print(f'processing ...:\n   {fp}')

        experiment_root = os.path.split(fp)
        im_path = os.path.join(experiment_root[0], "scan_001", "conc_0.00000", "00000.jpg")
        out = find_needle_pos(im_path)

        output.append(out)
        

for a,b in zip(file_path, output):
    print(f'{b["center"]}   {a}')

print("vvery veryvery veryvery veryvery veryvery veryery very")
# C:\Users\agaiosa\miniconda3\Scripts\conda run -n hld_ift1 python C:\Users\agaiosa\code\hld_ift_analysis_helpers\src\hld_ift_analysis_helpers\utils\visualize\find_roi_x_needle.py \\huckdfs-srv.science.ru.nl\huckdfs\RobotLab\Storage-Miscellaneous\aigars\temp\HLD_scan\AOT_OIW_C7_C16_v2 

