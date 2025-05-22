import sys 
sys.path.append("D:/projects/HLD_parameter_determination/hld_ift_analysis_helpers/src")

#from scripts.json_data_extraction import *
from hld_ift_analysis_helpers.montage_bits import *
from hld_ift_analysis_helpers.droplet_stats import *
from hld_ift_analysis_helpers.collect_files_folders import match_conc_at_end, collect_dirs

import argparse
from hld_ift_analysis_helpers.locations import data_json_path_to_exp_root_path, drop_stats_path

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
#parser.add_argument("-e", "--extraction_options", help = "a path to file that specifies extraction options: default value, value path in a data.json file, target variable name", default = "")
#parser.add_argument("-v", "--view", help = "view option analyzes input files and summarizes unique variable pathes in data.json file(s)", action = "store_true")
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


#================================================================================
# this part generates all output from all available droplets!!! takes a very long time

experiment_roots = list(map(data_json_path_to_exp_root_path, file_path))
drop_stats_csv_path = list(map(drop_stats_path, file_path))

for exp_root, path_out in zip(experiment_roots, drop_stats_csv_path):
    folders_use = match_conc_at_end(collect_dirs(exp_root))
    first = True
    if os.path.exists(path_out):
        os.remove(path_out)
    for p in folders_use:
        print(p)
        process_dir(p, path_out)
    
#================================================================================

