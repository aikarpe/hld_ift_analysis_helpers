import sys
sys.path.append("D:/projects/HLD_parameter_determination/hld_ift_analysis_helpers/src")

from hld_ift_analysis_helpers.montage_bits import *
from hld_ift_analysis_helpers.collect_files_folders import collect_data_jsons

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("source", help = "source of data.json file(s); can be a path to file, a path to file containing list of pathes or a folder")
parser.add_argument("-i", "--i_start", help = "first index to use for montage, default: 0", type = int, default = 0)
parser.add_argument("-n", "--n_images", help = "number of images per measurement to include, default: -1, all", type = int, default = -1)
parser.add_argument("-w", "--width", help = "width of image to include, default: 150 px", type = int, default = 150)
parser.add_argument("-t", "--test", help = "images to test, default: -1, all", type = int, default = -1)

args = parser.parse_args()

print(args.source)
print(args.i_start)
print(args.n_images)
print(args.width)
print(args.test)
exp_folders = []

def add_exp_folder(astr):
    if os.path.split(astr)[1] == "data.json":
        exp_folders.append(os.path.split(astr)[0])

if os.path.isfile(args.source):
    add_exp_folder(args.source)

if os.path.isdir(args.source):
    for f in collect_data_jsons(args.source):
        add_exp_folder(f)

if len(exp_folders) == 0:
    print("len is 0")
    print(f'No experiment found at path:\n   {args.source}')
if len(exp_folders) == 1:
    print("len is 1")
    print(f'processing ...:\n   {exp_folders[0]}')
    make_montage_of_measurement(
                        exp_folders[0],
                        i_start = args.i_start,
                        n_images = args.n_images,
                        roi_width = args.width, 
                        test = args.test)
if len(exp_folders) > 1:
    print("len is 2+")
    for ef in exp_folders:
        k = "y"
        #k = input(f'process? (y,n) {ef}')
        if k == "y":
            print(f'processing ...:\n   {ef}')
            make_montage_of_experiment(
                                        ef,
                                        i_start = args.i_start,
                                        n_images = args.n_images,
                                        roi_width = args.width, 
                                        test = args.test)





#-------- experiments individually --------------------
#make_montage_of_measurement("D:/temp_data/exp_2025-02-07_001_2.5g_BrijL4_C7-C16", i_start = 1, n_images = -1, roi_width = 150, test = -1)
#make_montage_of_measurement("D:/temp_data/exp_2025-02-13_001_5g_BrijL4_C7-C16", i_start = 1, n_images = -1, roi_width = 150, test = -1)
#make_montage_of_measurement("D:/temp_data/exp_2025-02-14_001_10g_BrijL4_C7-C16", i_start = 1, n_images = -1, roi_width = 150, test = -1)
#make_montage_of_measurement("D:/temp_data/exp_2025-02-24_001_20g_BrijL4_C7-C16", i_start = 1, n_images = -1, roi_width = 150, test = -1)
#make_montage_of_measurement("D:/temp_data/exp_2025-03-24_03g_BrijL4_C7C16_1", i_start = 1, n_images = -1, roi_width = 150, test = -1)
#make_montage_of_measurement("", i_start = 1, n_images = -1, roi_width = 150, test = -1)

#-------- all experiments at once --------------------
#make_montage_of_measurement("D:/temp_data", i_start = 1, n_images = 5, roi_width = 150, test = 5)
