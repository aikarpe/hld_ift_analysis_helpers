import sys 
#> sys.path.append("D:/projects/HLD_parameter_determination/hld_ift_analysis_helpers/src")

#from scripts.json_data_extraction import *
from hld_ift_analysis_helpers.montage_bits import *
from hld_ift_analysis_helpers.droplet_stats import *
from hld_ift_analysis_helpers.collect_files_folders import match_conc_at_end, collect_dirs, list_images

import argparse
from hld_ift_analysis_helpers.locations import (
            data_json_path_to_exp_root_path,
            drop_stats_path,
            data_json_path_to_drop_stats_output
            )

#>.......... #import pandas as pd
#>.......... #================================================================================
#>.......... # this part generates all output from all available droplets!!! takes a very long time
#>.......... 
#>.......... raw_data_folder = r'\\huckdfs-srv.science.ru.nl\huckdfs\RobotLab\Storage-Miscellaneous\aigars\temp\AOT_IB-45_C7C16_NaCl'
#>.......... #2>folders_use = match_conc_at_end(collect_dirs(raw_data_folder))
#>.......... 
#>.......... folders_use = list(filter(lambda x: re.search("exp_2025-04", x), 
#>..........                             match_conc_at_end(collect_dirs(raw_data_folder))))
#>.......... #--2-->folders_use = list(filter(lambda x: re.search("exp_2025-03-2", x), 
#>.......... #--2-->                            match_conc_at_end(collect_dirs(raw_data_folder))))
#>.......... #--1-->folders_use = list(filter(lambda x: re.search("03-24", x), 
#>.......... #--1-->                            match_conc_at_end(collect_dirs(raw_data_folder))))
#>.......... 
#>.......... #quick test |> folders_use = [folders_use[0]]
#>.......... 
#>.......... first = True
#>.......... path_out = os.path.join(raw_data_folder, "droplet_stats_data_2025_04_29.output.csv") #"data.output_temp.csv")
#>.......... if os.path.exists(path_out):
#>..........     os.remove(path_out)
#>.......... for p in folders_use:
#>..........     print(p)
#>..........     process_dir(p, path_out)
#>..........     
#>.......... #================================================================================

# a list of 
parser = argparse.ArgumentParser()
parser.add_argument("source", help = "source of data.json file(s); can be a path to file, a path to file containing list of pathes or a folder")
parser.add_argument("-r", "--raw_source", help = "file containing files to test", default = "")
parser.add_argument("-d", "--debug", help = "show debug images while processsing, use only with `raw_source`", action = "store_true", default = False)
#parser.add_argument("-e", "--extraction_options", help = "a path to file that specifies extraction options: default value, value path in a data.json file, target variable name", default = "")
#parser.add_argument("-v", "--view", help = "view option analyzes input files and summarizes unique variable pathes in data.json file(s)", action = "store_true")
parser.add_argument("-w", "--width", help = "width of image to include, default: 150 px", type = int, default = 150)
parser.add_argument("-m", "--max_weight", help = "max weight, default: 0.8", type = float , default = 0.8)
parser.add_argument("-1", "--save_raw_image", help = "save original image of area under needle", action = "store_true")
parser.add_argument("-2", "--save_object_image", help = "save processed dripping image of area under needle", action = "store_true")
parser.add_argument("-a", "--save_raw_and_object_image", help = "save original and processed dripping image of area under needle", action = "store_true")
parser.add_argument("-t", "--test", help = "test few images from each measurement", type = int, default = -1)

args = parser.parse_args()

fl_save_raw_im = args.save_raw_image or args.save_raw_and_object_image
fl_save_dripping_im = args.save_object_image or args.save_raw_and_object_image

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



#================================================================================
# this part generates all output from all available droplets!!! takes a very long time


experiment_roots = list(map(data_json_path_to_exp_root_path, file_path))
drop_stats_csv_path = list(map(drop_stats_path, file_path))

for exp_root, path_out, data_json_path in zip(experiment_roots, drop_stats_csv_path, file_path):

    root = data_json_path_to_drop_stats_output(data_json_path)

    def fn_proc_im(apath): 
        return process_dripping_stats(
                apath,
                root, 
                width = args.width, 
                max_weight = args.max_weight, 
                save_raw_region = fl_save_raw_im, 
                save_object_image = fl_save_dripping_im
                )


    folders_use = match_conc_at_end(collect_dirs(exp_root))
    first = True
    if os.path.exists(path_out):
        os.remove(path_out)
    for p in folders_use:
        print(p)
        process_dir(p, path_out, fn = fn_proc_im, test = args.test)
    
#================================================================================

