import sys 
sys.path.append("D:/projects/HLD_parameter_determination/hld_ift_analysis_helpers/src")

import json
#from scripts.json_data_extraction import *
from hld_ift_analysis_helpers.montage_bits import *
from hld_ift_analysis_helpers.droplet_stats import *
from hld_ift_analysis_helpers.collect_files_folders import  collect_data_jsons

#import pandas as pd
#   keeps track of processed experiments
#        json config file
#            [
#                {
#                    experiment_name: {
#                        data_extracted: false,
#                        exp_montage: false,
#                        meas_montage: false,
#                        stats: false,
#
#                    }
#                },
#            ]
#    read config file
#    checks if new experiment is acquired
#    add new experiments if any
#    for exp_params in config:
#        if not data_extracted:
#            try:
#                extract
#                change config flag
#            except:
#                somethig failed
#                report to log file
#        if not exp_montage:
#            do exp montage
#        if not meas_montage:
#            do meas montage
#        if not stats:
#            do stats
#
#    would need config file; extraction options, processing_log_file
#
root = r'\\huckdfs-srv.science.ru.nl\huckdfs\RobotLab\Storage-Miscellaneous\aigars\temp\AOT_IB-45_C7C16_NaCl'
proc_config_folder = ".processing_config"

log_file_path = os.path.join(root, proc_config_folder, "processing_log.log")
config_file_path = os.path.join(root, proc_config_folder, "current_processing_status.json")
extraction_options_path = os.path.join(root, proc_config_folder, "data.json_extraction_options.json")

print(f'all relevant pathes:\nroot: {root}\nlog_file: {log_file_path}\nconfig_file: {config_file_path}\nextraction_opt: {extraction_options_path}')

# configuration: processing status
try:
    with open(config_file_path, "r") as f:
        config = json.load(f)
    with open(f'{config_file_path}.bak', "w") as bakf:
        json.dump(config, bakf, indent = 2)
except: 
    config = dict()

# extraction options
to_extract = None
try:
    with open(extraction_options_path, "r") as f:
        to_extract = pd.DataFrame.from_dict(
                json.load(f),
                orient = "index",
                columns = ['default', 'generic_path', 'new_names'])
except: 
    print(f'you need to define extractions options!!! They should be at locatin:\n  {extraction_options_path}')
    exit()


def format_error_message(source, msg):
    return f'{msg}: {source}'

def log_error_message(source, msg):
    a_mode = "a" if os.path.exists(log_file_path) else "w"
    with open(log_file_path, a_mode) as f:
        f.write(format_error_message(source, msg))

        
steps = [ dict(
               step_name = "data_extraction",
               #fn = lambda source: extract_experiment_data(source, to_extract, save_data = True, save_extraction_pathes = True),
               fn = lambda source: print(f"extracting data!!! {source}"),
               err_msg = "failed data extraction"
               ),
          dict(
               step_name = "exp_montage",
               fn = lambda source: print(f"making exp montage!!! {source}"),
               err_msg = "failed to create experimental montage", 
               ),
          dict(
               step_name = "meas_montage",
               fn = lambda source: print(f"making meas montage!!! {source}"),
               err_msg = "failed to create measuement montages", 
               ),
          dict(
               step_name = "stats",
               fn = lambda source: print(f"making stats!!! {source}"),
               err_msg = "failed to create stats",
               )
]

def add_missing_step_to_config():
    step_names = list(map(lambda x: x["step_name"], steps))
    for exp in config:
        keys_in_experiment = config[exp].keys()
        for a_key in step_names:
            if a_key not in keys_in_experiment:
                config[exp][a_key] = True

def add_missing_exp_to_config():
    data_jsons = collect_data_jsons(root)
    available_experiments = list(map(lambda x: os.path.split(os.path.split(x)[0])[1], data_jsons))

    data_jsons_dict = dict()
    for exp, p in zip(available_experiments, data_jsons):
        data_jsons_dict[exp] = p

    exp_in_config = config.keys()
    for exp in available_experiments:
        if exp not in exp_in_config:
            config[exp] = dict()
            config[exp]["data_file"] = data_jsons_dict[exp]

add_missing_exp_to_config()
add_missing_step_to_config()

for exp in config:
    for a_step in steps:
        step_name = a_step["step_name"]
        source = config[exp]["data_file"]
        if not config[exp][step_name]:
            try:
                a_step["fn"](source)
                config[exp][step_name] = True
            except:
                log_error_message(source, a_step["err_msg"])

with open(config_file_path, "w") as f:
    json.dump(config, f, indent = 2)

