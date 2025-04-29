import sys
sys.path.append("D:/projects/HLD_parameter_determination/hld_ift_analysis_helpers/src")

import json
from hld_ift_analysis_helpers.json_data_extraction import *
from hld_ift_analysis_helpers.collect_files_folders import collect_data_jsons
import pandas as pd
import argparse


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("source", help = "source of data.json file(s); can be a path to file, a path to file containing list of pathes or a folder")
parser.add_argument("-e", "--extraction_options", help = "a path to file that specifies extraction options: default value, value path in a data.json file, target variable name", default = "")
parser.add_argument("-v", "--view", help = "view option analyzes input files and summarizes unique variable pathes in data.json file(s)", action = "store_true")
args = parser.parse_args()

# select source files
file_path = []
extraction_options = args.extraction_options if os.path.isfile(args.extraction_options) else ""

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

    if extraction_options == "":
        options_files = collect_files(args.source, "data.json_extraction_option.json$")
        N = len(options_files)
        if N > 1:
            print("multiple options files found:")
            for i,f in enumerate(options_files):
                print(f'{i+1: 3d>}: {f}') 
            k = input("select file to use")
            try:
                index = int(k) - 1
            except:
                index = 0
            finally:
                print(f'using {options_files[index]} for options')
                extraction_options = options_files[index]
        if N == 1:
            extraction_options = options_files[0]

if len(file_path) == 0:
    print(f'not sure what to do with given source:\n `{args.source}`\n exiting')
    exit


to_extract = None
if extraction_options != "":
    with open(extraction_options, "r") as f:
        to_extract = pd.DataFrame.from_dict(
                json.load(f),
                orient = "index",
                columns = ['default', 'generic_path', 'new_names'])


view_keys = args.view
extract = not view_keys
for p in file_path:
    print(p)
    if view_keys:
        with open(p, 'r', encoding='utf-8') as f:
            data_dict = json.load(f)
        json_path = json_tree(data_dict)
        zz = sorted(list(set(json_path)))
        zz = list(map(lambda x: re.sub("__index__", "_i_", x), zz))
        #zx = list(map(lambda x: re.sub("\\/", "\\\\", x), 
        #                list(map(lambda x: x + "            <---", to_extract['generic_path']))))



        print(f'=================== {p}')
        #for x in sorted(zz + zx):
        for x in sorted(zz):
            print(x)

        if to_extract is not None:
            matching_path = []
            unmatched_path = []
            for a_path in list(map(lambda x: re.sub("\\/", "\\\\", x), to_extract['generic_path'])):
                if a_path in zz:
                    matching_path.append(a_path)
                else:
                    unmatched_path.append(a_path)

            print(f'=================== {p}')

            print(f'                    matching path: {len(matching_path)}')
            for x in matching_path:
                print(x)

            print(f'                    unmatched path: {len(unmatched_path)}')
            for x in unmatched_path:
                print(x)

    if extract:
        if extraction_options == "":
            print(f'not sure how to extract parameters, no extraction options specified or found\n exiting')
            exit
        extract_experiment_data(p, to_extract, save_data = True, save_extraction_pathes = True)

