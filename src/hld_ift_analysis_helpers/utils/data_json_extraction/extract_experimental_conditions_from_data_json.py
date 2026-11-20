import sys
#> sys.path.append("D:/projects/HLD_parameter_determination/hld_ift_analysis_helpers/src")

import json
from hld_ift_analysis_helpers.json_data_extraction import *
from hld_ift_analysis_helpers.collect_files_folders import collect_data_jsons
from hld_ift_analysis_helpers.locations import (
                                    extracted_data_source_path,
                                    extracted_data_path,
                                    data_json_path_to_processed_data_json_path 
                                    )
import pandas as pd
import argparse
import functools
from tkinter.filedialog import askopenfilename
#from tkinter.filedialog import askopenfilename, askdirectory

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("source", help = "source of data.json file(s); can be a path to file, a path to file containing list of pathes or a folder")
parser.add_argument("-s", "--settings_file", help = "a path to settings file: default value, ``", default = "")
parser.add_argument("-w", "--weight_fraction", help = "if used, a weight fraction for each component is reported, default value `false` - mass concentration per volume is reported (g/ml)", action = "store_true")
parser.add_argument("-o", "--output_file", help = "a path to an output file where composition data is saved, default value, `` (file is saved @`<path_to_data.json>/data.json.components.csv`)", default = "")
args = parser.parse_args()

# select source files
file_path = []

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

if os.path.isfile(args.settings_file):
    with open(args.settings_file, "r") as f:
        data = json.load(f)
    filename = askopenfilename(title='Select data.json file ...', defaultextension = "json", initialdir = data['DATA_PATH'])
    
    if os.path.isfile(filename):
        file_path.append(filename)

if len(file_path) == 0:
    print(f'not sure what to do with given source:\n `{args.source}`\n exiting')
    exit()


def parse_scan(data,scan, fn_meas):
    if 'measurements' in scan.keys():
        return list(set(data + functools.reduce(fn_meas, scan['measurements'], [])))
    else:
        return data

def parse_measurement(data, meas, label):
    if label in meas.keys():
        return list(set(data + parse_solution(meas[label])))
    else:
        return data

def parse_solution(sol):
    if 'components' in sol:
        return list(sol['components'].keys())


def extract_scan(data, scan):
    scan_lbl = scan['label']
    def update_dict(adct):
        adct['scan'] = scan_lbl
        return adct
    
    out = list()
    if 'measurements' in scan and scan['measurements'] is not None and len(scan['measurements']) > 0:
        return data + list(map(update_dict, extract_measurements(scan['measurements'])))
    else:
        return data

def extract_solution(sol, modifier, keys, adict):
    new_lbl = lambda x: f'{modifier}_{x}'
    adict[new_lbl('ro')] = sol['ro']
    for a_key in keys:
        if a_key in sol['components']:
            adict[new_lbl(a_key)] = sol['components'][a_key]['w']
        else:
            adict[new_lbl(a_key)] = 0

def components_starting_with_substr(a_lst, substr):
    return [ x for x in a_lst if x != f'{substr}ro' and x.startswith(substr)]

def w_components_to_mv(df, substr):
    lbls = components_starting_with_substr(list(df), substr)
    print(lbls)
    for lbl in lbls:
        df[f'{lbl}_mv'] = df[lbl] * df[f'{substr}ro']

for afile in file_path:

    with open(afile, "r") as f:
        data = json.load(f)
    
    scans = data['scans']
    
    inn_fn = lambda x,y: parse_measurement(x,y,'solution_inner')
    cmp_inner = functools.reduce(lambda x,y: parse_scan(x,y, inn_fn), scans, [])
    
    out_fn = lambda x,y: parse_measurement(x,y,'solution_outer')
    cmp_outer = functools.reduce(lambda x,y: parse_scan(x,y, out_fn), scans, [])

    def extract_measurements(measurements):
        out = list()
        for meas in measurements:
            if 'label' in meas and \
                'solution_inner' in meas and \
                'solution_outer' in meas and \
                'autofocus' not in meas['label']:
                a_dict = dict()
                a_dict['measurement'] = meas['label']
                extract_solution(meas['solution_inner'], 'si', cmp_inner, a_dict)
                extract_solution(meas['solution_outer'], 'so', cmp_outer, a_dict)
                out.append(a_dict)
        return out


    #extract_measurements(scans[1]['measurements'])
                
    output = list(functools.reduce(extract_scan, scans, []))
    
    cdf = pd.DataFrame(output)
   
    if not args.weight_fraction: 
        print("will drop")
        si_calc = components_starting_with_substr(list(cdf), "si_")
        so_calc = components_starting_with_substr(list(cdf), "so_")

        w_components_to_mv(cdf, "si_")
        w_components_to_mv(cdf, "so_")

        cdf = cdf.drop(columns = si_calc + so_calc)
        print(si_calc + so_calc)

    print(afile)
    print(cdf.to_markdown())
    
    def component_table(df, param_lbl): 
        unique_scans = list(set(cdf["scan"].tolist()))
        unique_scans.sort()
        unique_concs = list(set(cdf["measurement"].tolist()))
        unique_concs.sort()
        def extr_value(scan_val, meas_val, col_lbl, df):
            temp_df = df[ (df.scan == scan_val) & (df.measurement == meas_val) ]
            value = "." if len(temp_df.index) < 1 else str(temp_df[col_lbl].tolist()[0])
            return value

        index = unique_scans
        data = dict()
        for conc in unique_concs:
            data[conc] = list(map(lambda x: extr_value(x, conc, param_lbl, df), unique_scans))
        return pd.DataFrame(data, index = index)

    columns = [x for x in list(cdf) if x not in ['scan', 'measurement']]

    for acol in columns:
        print("--------------------------------------------------------------------------------")
        print("")
        print(f'### {acol}')
        print("")
        print(component_table(cdf, acol).to_markdown())
        print("")
        print("--------------------------------------------------------------------------------")


    output_file = afile + ".components.csv" if args.output_file == "" else args.output_file
    cdf.to_csv(output_file, sep = ",", encoding = 'utf-8')
    print(f'composition data written at:\n   {output_file}')

print("******** done ********")
