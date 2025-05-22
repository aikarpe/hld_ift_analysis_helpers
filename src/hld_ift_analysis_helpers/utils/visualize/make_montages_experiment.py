import sys
sys.path.append("D:/projects/HLD_parameter_determination/hld_ift_analysis_helpers/src")


from hld_ift_analysis_helpers.montage_bits import *
from hld_ift_analysis_helpers.collect_files_folders import collect_data_jsons
from hld_ift_analysis_helpers.locations import data_json_path_to_exp_montage_path

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("source", help = "source of data.json file(s); can be a path to file, a path to file containing list of pathes or a folder")
parser.add_argument("-i", "--i_start", help = "first index to use for montage, default: 0", type = int, default = 0)
parser.add_argument("-n", "--n_images", help = "number of images per measurement to include, default: 5", type = int, default = 5)
parser.add_argument("-w", "--width", help = "width of image to include, default: 150 px", type = int, default = 150)
parser.add_argument("-t", "--test", help = "images to test, default: -1, all", type = int, default = -1)
parser.add_argument("-o", "--output_path", help = "output path for a montage", type = str, default = "")

args = parser.parse_args()

print(args.source)
print(args.i_start)
print(args.n_images)
print(args.width)
print(args.test)
print(args.output_path)

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


for fp in file_path:
    k = "y"
    #k = input(f'process? (y,n) {fp}')
    if k == "y":
        print(f'processing ...:\n   {fp}')
        if args.output_path == "":
            montage_output_path = data_json_path_to_exp_montage_path(fp)
            os.makedirs(os.path.split(montage_output_path)[0], exist_ok = True)
        else:
            montage_output_path = args.output_path
        make_montage_of_experiment(
                                    os.path.split(fp)[0],
                                    i_start = args.i_start,
                                    n_images = args.n_images,
                                    roi_width = args.width, 
                                    test = args.test,
                                    output_path = montage_output_path)





