print("111111111111111")
import sys
#> sys.path.append("D:/projects/HLD_parameter_determination/hld_ift_analysis_helpers/src")
print("111111111111111")


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("source", help = "source of data.json file(s); can be a path to file, a path to file containing list of pathes or a folder")
parser.add_argument("-i", "--i_start", help = "first index to use for montage, default: 0", type = int, default = 0)
parser.add_argument("-n", "--n_images", help = "number of images per measurement to include, default: 5", type = int, default = 5)
parser.add_argument("-w", "--width", help = "width of image to include, default: 150 px", type = int, default = 150)
parser.add_argument("-t", "--test", help = "images to test, default: -1, all", type = int, default = -1)
parser.add_argument("-o", "--output_path", help = "output path for a montage", type = str, default = "")
parser.add_argument("-v", "--flip_variables", help = "flips scan variable and scans", type = bool, default = False)
parser.add_argument("-c", "--flip_conc_order", help = "reverse ordering of scan variable", type = bool, default = False)
parser.add_argument("-s", "--flip_scan_order", help = "reverse ordering of scans", type = bool, default = False)
parser.add_argument("-x", "--roi_x_start", help = "x coordinate for beginning of needle roi, default: -1, needle roi determined automatically", type = int, default = -1)
parser.add_argument("-e", "--csv_file", help = "csv file as an input for montage", type = bool, default = False)


args = parser.parse_args()
print("111111111111111")

print(args.source)
print(args.i_start)
print(args.n_images)
print(args.width)
print(args.test)
print(args.output_path)
print(args.flip_variables)
print(args.flip_conc_order)
print(args.flip_scan_order)

#from hld_ift_analysis_helpers.montage_bits import *
#import hld_ift_analysis_helpers.montage_bits 
from hld_ift_analysis_helpers.montage_bits import make_montage_of_experiment, make_montage_of_experiment_csv
from hld_ift_analysis_helpers.collect_files_folders import collect_data_jsons
from hld_ift_analysis_helpers.locations import data_json_path_to_exp_montage_path
print("111111111111111")

# select source files
file_path = []
#extraction_options = args.extraction_options if os.path.isfile(args.extraction_options) else ""

def process_string_pointing_to_data_json_file(astr):
    if os.path.split(astr)[1] == "data.json":
        file_path.append(astr)

print(args.csv_file)
print(args.source)

if args.csv_file:
    montage_output_path = args.output_path if args.output_path != "" else f'{args.source}.jpg'
    make_montage_of_experiment_csv(
                            args.source,
                            i_start = args.i_start,
                            n_images = args.n_images,
                            roi_width = args.width, 
                            test = args.test,
                            output_path = montage_output_path,
                            reverse_measurement_order = args.flip_conc_order, 
                            reverse_scan_order = args.flip_scan_order,
                            transpose_scan_measurement = args.flip_variables,
                            roi_start = args.roi_x_start
                            )
    print("Done making montage, will exit now!")
    exit()

    
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
                                    output_path = montage_output_path,
                                    reverse_measurement_order = args.flip_conc_order, 
                                    reverse_scan_order = args.flip_scan_order,
                                    transpose_scan_measurement = args.flip_variables,
                                    roi_start = args.roi_x_start
                                    )





