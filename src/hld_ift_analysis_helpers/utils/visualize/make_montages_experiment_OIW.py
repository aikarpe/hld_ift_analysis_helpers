import sys
#> sys.path.append("D:/projects/HLD_parameter_determination/hld_ift_analysis_helpers/src")

import pandas as pd
from functools import reduce 
import itertools

from hld_ift_analysis_helpers.montage_bits import *
from hld_ift_analysis_helpers.collect_files_folders import collect_data_jsons
from hld_ift_analysis_helpers.locations import data_json_path_to_exp_montage_path, montage_name__experiment

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("source", help = """
    path to csv data file containing following columns
        - path: String, path to image
        - experiment: String, experiment name (also folder name)
        - experiment_root: String, path to experiment root
        - scan: String, scan name of type "scan_<000>" where <000> is scan index
        - concentration: String, measurement label of type "conc_<x.xxxxx>"
        - concentration_val: float, relative concentration of a scan (float of "<x.xxxxx>")
        - image_name: String: file name of type "<XXXXX>.jpg", were XXXXX is index
        - image_index: int, image index in measurement
    """
    )
parser.add_argument("-i", "--i_start", help = "first index to use for montage, default: 0", type = int, default = 0)
parser.add_argument("-n", "--n_images", help = "number of images per measurement to include, default: 5", type = int, default = 5)
parser.add_argument("-w", "--width", help = "width of image to include, default: 150 px", type = int, default = 150)
parser.add_argument("-t", "--test", help = "images to test, default: -1, all", type = int, default = -1)
parser.add_argument("-o", "--output_folder", help = "output folder for a montages, default: csv_file folder", type = str, default = "")
parser.add_argument("-v", "--flip_variables", help = "flips scan variable and scans", type = bool, default = False)
parser.add_argument("-c", "--flip_conc_order", help = "reverse ordering of scan variable", type = bool, default = False)
parser.add_argument("-s", "--flip_scan_order", help = "reverse ordering of scans", type = bool, default = False)
parser.add_argument("-a", "--same_format", help = "apply uniform shape of all montages", type = bool, default = True)

args = parser.parse_args()

print(args.source)
print(args.i_start)
print(args.n_images)
print(args.width)
print(args.test)
print(args.output_folder)
print(args.flip_variables)
print(args.flip_conc_order)
print(args.flip_scan_order)
print(args.same_format)

#> """
#> !!!ADD option to reverse scans, concentration, flip scans with concentration!!!
#> 
#> read source (panda file)
#> ensure that it contains all the columns needed
#> """

#> def process_string_pointing_to_data_json_file(astr):
#>     if os.path.split(astr)[1] == "data.json":
#>         file_path.append(astr)
#> 
#> if os.path.isfile(args.source):
#>     if os.path.split(args.source)[1] == "data.json":
#>         # a single source file
#>         process_string_pointing_to_data_json_file(args.source) 
#>     else:
#>         # a file containing list of source files
#>         with open(args.source, 'r') as file:
#>             for line in file:
#>                 process_string_pointing_to_data_json_file(line) 
#> 
#> if os.path.isdir(args.source):
#>     for f in collect_data_jsons(args.source):
#>         process_string_pointing_to_data_json_file(f)
#> 
#> if len(file_path) == 0:
#>     print(f'not sure what to do with given source:\n `{args.source}`\n exiting')
#>     exit()

def add_unique_conditions(df):
    print("add_unique_conditions: enter")
    def unique_sorted(name):
        return sorted(list(set(df[name])))

    u_scans = unique_sorted("scan")
    u_exp = unique_sorted("experiment")
    u_conc = unique_sorted("concentration")
    u_im_names = unique_sorted("image_name")

    all_combinations = [ x for x in itertools.product(u_exp, u_scans, u_conc)]

    splits = {g:df for g,df in df.groupby(["experiment","scan","concentration"])}
    available_combs = list(splits.keys())

    missing_combs = list(set(all_combinations).difference(set(available_combs)))

    dfs = [df]
    for exp_v, scan_v, conc_v in missing_combs:
        dftemp = pd.DataFrame({'image_name': u_im_names})    
        dftemp['path'] = ""
        dftemp['experiment'] = exp_v
        dftemp['scan'] = scan_v
        dftemp['concentration'] = conc_v
        dftemp['experiment_root'] = ""
        dfs = dfs + [dftemp]

    df_out = pd.concat(dfs)
    print("add_unique_conditions: exit")
    return df_out
  
csv_folder = os.path.split(args.source)[0]
csv_folder = "C:/Users/aigar/delme"

try:
    df = pd.read_csv(args.source)
except Exception as e:
    print("failed while opening csv file at:")
    print(args.source)
    print(e)
    print(".....terminating")
    exit()

if args.same_format:
    df_2 = add_unique_conditions(df)
else:
    df_2 = df

exp_split = {g:df for g,df in df_2.groupby("experiment")}    

print(list(exp_split.keys()))

for an_exp_key in exp_split.keys():
    print(an_exp_key)
    print("loop begins")
    if not args.same_format:
        df_3 = add_unique_conditions(exp_split[an_exp_key])
    else:
        df_3 = exp_split[an_exp_key]
    

    def unique_sorted(name, df):
        return sorted(list(set(df[name])))
    u_scans = unique_sorted("scan", df_3)
    u_exp = unique_sorted("experiment", df_3)
    u_conc = unique_sorted("concentration", df_3)
    u_im_names = unique_sorted("image_name", df_3)
    print("experiments")
    print(u_exp)
    print("scans")
    print(u_scans)
    print("concentrations")
    print(u_conc)
    print("image_names")
    print(u_im_names)

    df_3.sort_values(
        by = ['experiment','scan', 'concentration', 'image_name'],
        ascending = [True, not args.flip_scan_order, not args.flip_conc_order, True],
        inplace = True,
        ignore_index = True
        )


    print("will make montage nowish ....") 
    print(df_3['experiment'].to_list()[0])
    print(os.path.join(csv_folder, montage_name__experiment( 
        df_3['experiment'].to_list()[0], 
        args.i_start,
        args.n_images,
        args.width)))
    make_montage_of_experiment_df(
                                df_3,
                                i_start = args.i_start,
                                n_images = args.n_images,
                                roi_width = args.width, 
                                test = args.test,
                                output_path = os.path.join(
                                                    csv_folder, 
                                                    montage_name__experiment(
                                                                    df_3['experiment'].to_list()[0], 
                                                                    args.i_start,
                                                                    args.n_images,
                                                                    args.width)),
                                reverse_measurement_order = args.flip_conc_order, 
                                reverse_scan_order = args.flip_scan_order,
                                transpose_scan_measurement = args.flip_variables
                                )
                                
    print("loop ends=======================================================")

#> """
#> df --> add_unique_conditions --> split --> order --> montage
#> df --> split --> add_unique_conditions --> order --> montage
#> if args.same_format:
#>     get
#>         unique scans
#>         unique concentrations
#>         unique experiments
#>         unique image_names
#>     find combinations of (experiment, scan, concentratio) that is not represented in df
#> 
#>     for each combination (an_exp, a_scan, a_conc):
#>         make data frame of all image_names
#>         add path equal to ""
#>         add experiment <- an_exp
#>         add scan <- a_scan
#>         add conc <- a_conc
#> 
#> split dataFrame by experiment && sort by 
#>     experiment
#>     scans (reverse if args.flip_scan_order)
#>     conc (reverse if args.flip_conc_order)
#>     image_namejjj
#> 
#> for each experiments df:
#>     create montage(s)
#>         flip_variables
#> 
#> 
#> 
#> for fp in file_path:
#>     k = "y"
#>     #k = input(f'process? (y,n) {fp}')
#>     if k == "y":
#>         print(f'processing ...:\n   {fp}')
#>         if args.output_path == "":
#>             montage_output_path = <replace for a standard exp_montage name in csv_path folder,
#>                                     considering that df can contain multiple experiments
#>                                     data_json_path_to_exp_montage_path(fp) >
#>             os.makedirs(os.path.split(montage_output_path)[0], exist_ok = True)
#>         else:
#>             montage_output_path = args.output_path
#>         make_montage_of_experiment(
#>                                     os.path.split(fp)[0],
#>                                     i_start = args.i_start,
#>                                     n_images = args.n_images,
#>                                     roi_width = args.width, 
#>                                     test = args.test,
#>                                     output_path = montage_output_path)
#> 
#> 
#> 
#> cs <- "//huckdfs-srv.science.ru.nl/huckdfs/RobotLab/Storage-Miscellaneous/aigars/temp/HLD_scan/Ecosurf_EH-3_OIW_v2"
#> my_path <- function(x) do.call(path, as.list(x))
#> temptemp2 <- temptemp %>%
#>     mutate(
#>         concentration = measurement,
#>         image_name = path %>% path_split %>% map_chr(.f = ~ .x[[length(.x)]]),
#>         path = path %>% path_split %>% map_chr(.f = function(x) {
#>                                                         n <- length(x)
#>                                                         my_path(c(cs, x[seq(n - 3, n)]))
#>                                                         }),
#>         ) %>%
#>     select("path", "experiment", "scan", "concentration", "image_name")
#> 
#> """
#> 
#> 

