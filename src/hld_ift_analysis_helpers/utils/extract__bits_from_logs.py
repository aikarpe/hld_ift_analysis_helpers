
#python D:/temp_data/extract__bits_from_logs.py
import sys 
import functools
sys.path.append("D:/projects/HLD_parameter_determination/hld_ift_analysis_helpers/src")
#sys.path.append("C:/Users/nguye/Desktop/Radboud/hld_code/hld_ift_analysis_helpers/src")
import os
#from scripts.json_data_extraction import *
from hld_ift_analysis_helpers.json_data_extraction import *
from hld_ift_analysis_helpers.locations import (
                                        data_json_path_to_raw_log,
                                        data_json_path_to_log_compact_ops,
                                        data_json_path_to_log_keys,
                                        data_json_path_to_log_asp_disp_table,
                                        data_json_path_to_log_urls
                                        )
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("source", help = "source of data.json file(s); can be a path to file, a path to file containing list of pathes or a folder")
#parser.add_argument("-e", "--extraction_options", help = "a path to file that specifies extraction options: default value, value path in a data.json file, target variable name", default = "")
parser.add_argument("-v", "--view_keys", help = "view available keys in data.json file", action = "store_true")
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


EXTRACT_URLS = 1                # extract urls for each logged command
VIEW_KEYS = 2                   # all keys
EXTRACT_ASP_DISP = 4            # use with  extract__bits_from_logs__plot_r  r script!!!
EXTRACT_COMPACT_OPERATIONS = 8  # all commands in compact form

operations_requested = [VIEW_KEYS] if args.view_keys else [EXTRACT_URLS, EXTRACT_ASP_DISP, EXTRACT_COMPACT_OPERATIONS]

to_extract = [ 
        ["",          "input/data",                                          "input_data"],
        ["",          "output/data/commandType",                             "commandType"],
        ["",          "output/data/completedAt",                             "completedAt"],
        ["",          "output/data/createdAt",                               "createdAt"],
        ["",          "output/data/id",                                      "id"],
        ["",          "output/data/intent",                                  "intent"],
        ["",          "output/data/key",                                     "key"],
        ["",          "output/data/notes",                                   "notes"],
        [-1000,       "output/data/params/flowRate",                      "flowRate"],
        ["",          "output/data/params/labwareId",                        "labwareId"],
        ["",          "output/data/params/pipetteId",                        "pipetteId"],
        [-1000,       "output/data/params/volume",                        "volume"],
        [-1000,       "output/data/params/wellLocation/offset/x",         "offset_x"],
        [-1000,       "output/data/params/wellLocation/offset/y",         "offset_y"],
        [-1000,       "output/data/params/wellLocation/offset/z",         "offset_z"],
        ["",          "output/data/params/wellLocation/origin",              "origin"],
        [-1000,       "output/data/params/wellLocation/volumeOffset",     "volumeOffset"],
        ["",          "output/data/params/wellName",                         "wellName"],
        [-1000,       "output/data/result/position/x",                    "position_x"],
        [-1000,       "output/data/result/position/y",                    "position_y"],
        [-1000,       "output/data/result/position/z",                    "position_z"],
        [-1000,       "output/data/result/volume",                        "r_volume"],
        ["",          "output/data/startedAt",                               "startedAt"],
        ["",          "output/data/status",                                  "status"],
        ["",          "time",                                                "time"],
        ]



def extr_fn(x):
    a_dict = dict()
    for inst in to_extract:
        a_dict[inst[2]] = get_element(x, inst[1], inst[0]) 
    return a_dict

column_headings = list(map(lambda x: x[2], to_extract))

def modify_path(path, appendix):
    return  path + appendix
def load_data(p):
    data_dict = None
    with open(p, 'r', encoding='utf-8') as f:
        data_dict = json.load(f)
    return data_dict
def input_url(command_dict):
    inputs = command_dict["input"]
    return inputs["url"] if type(inputs) == dict and "url" in inputs.keys() else "na"
def is_command(command_dict):
    command_url = input_url(command_dict)
    cmp = command_url.split("/")
    return cmp[len(cmp) - 1] == "commands"
def command_entry_to_string(command_dict, translate = None):
    """
    returns string of type:
    {timestamp}/{command}/{status}/{pipette}/{slot or labware id}/{well}/{volume}
    """
    get_pipette_loc = get_pipette if translate is None else lambda x: translate["pipette_to_human"](get_pipette(x))
    get_labware_loc = get_labware_id if translate is None else lambda x: translate["labware_to_human"](get_labware_id(x))
    components = list(map(lambda x: x(command_dict),
                            [get_time, 
                                get_status, 
                                get_command, 
                                get_pipette_loc, 
                                get_labware_loc,
                                get_well                        ]))
    if components[2] in ["aspirate", "dispense"]:
        components.append(get_volume(command_dict))

    return functools.reduce(lambda x,y: str(x) + "/" + str(y), components)

def get_time(entry_dict):
    return entry_dict["time"] 
def get_command(command_dict):
    return command_dict["output"]["data"]["commandType"] if is_command(command_dict) else "not_a_command"
def get_status(command_dict):
    return command_dict["output"]["data"]["status"] if is_command(command_dict) else "not_available"
def get_pipette(command_dict):
    if is_command(command_dict) and "pipetteId" in command_dict["output"]["data"]["params"].keys():
        return command_dict["output"]["data"]["params"]["pipetteId"] 
    return "not_available"
def get_labware_id(command_dict):
    if is_command(command_dict) and "labwareId" in command_dict["output"]["data"]["params"].keys():
        return command_dict["output"]["data"]["params"]["labwareId"] 
    return "not_available"
def get_well(command_dict):
    if is_command(command_dict) and "wellName" in command_dict["output"]["data"]["params"].keys():
        return command_dict["output"]["data"]["params"]["wellName"] 
    return "not_available"
def get_volume(command_dict):
    if is_command(command_dict) and get_command(command_dict) in ["aspirate", "dispense"]:
        return command_dict["output"]["data"]["params"]["volume"] 
    else:
        return ""

def convertion_functions(a_data_dict):
    run_url = ""
    for el in a_data_dict:
        if is_command(el):
            url = input_url(el)
            run_url = os.path.split(url)[0]
        if run_url != "" and "data" in el["output"] and type(el["output"]["data"]) == dict:
            a_data = el["output"]["data"]
            data_keys = a_data.keys()
            if  input_url(el) == run_url and "pipettes" in data_keys and "labware" in data_keys: 

                id_to_hum_p = dict()
                hum_to_id_p = dict()
                for lw in a_data["pipettes"]:
                    an_id = lw["id"]
                    a_mount = lw["mount"]
                    id_to_hum_p[an_id] = a_mount
                    hum_to_id_p[a_mount] = an_id

                id_to_hum_l = dict()
                hum_to_id_l = dict()
                for lw in a_data["labware"]:
                    an_id = lw["id"]
                    a_slot = lw["location"]["slotName"]
                    id_to_hum_l[an_id] = a_slot
                    hum_to_id_l[a_slot] = an_id

                def translate_value(a_dict, val):
                    return a_dict[val] if val in a_dict.keys() else val

                print(f' for translation:\n{id_to_hum_p}\n{id_to_hum_l}\n{hum_to_id_p}\n{hum_to_id_l}')
                return dict(
                    pipette_to_human = lambda x: translate_value(id_to_hum_p, x),
                    labware_to_human = lambda x: translate_value(id_to_hum_l, x),
                    pipette_to_id = lambda x: translate_value(hum_to_id_p, x),
                    labware_to_id = lambda x: translate_value(hum_to_id_l, x)
                    )
    return None

        

#action = [ EXTRACT_URLS, EXTRACT_ASP_DISP, VIEW_KEYS, EXTRACT_COMPACT_OPERATIONS ]
action = operations_requested

for p in file_path:
    print(f'path found:\n{p}')
    data_dict = load_data(data_json_path_to_raw_log(p))

    log_folder = os.path.split(data_json_path_to_log_keys(p))[0]
    os.makedirs(log_folder, exist_ok = True)

    if EXTRACT_URLS in action:
        with open(data_json_path_to_log_urls(p), 'w') as f:
            for el in data_dict:
                f.write(f"{input_url(el)}\n") 

    if VIEW_KEYS in action:
        json_path = json_tree(data_dict)
        zz = sorted(list(set(json_path)))
        zz = list(map(lambda x: re.sub("__index__", "_i_", x), zz))
        with open(data_json_path_to_log_keys(p), 'w') as f:
            for el in zz:
                f.write(f"{el}\n") 

    if EXTRACT_ASP_DISP in action:
        useful = list(filter(lambda x: get_element(x, "output/data/commandType", "") in ["aspirate", "dispense"], data_dict))
        json_path = json_tree(useful)


        print(f'=================== {p}')
        zz = sorted(list(set(json_path)))
        zz = list(map(lambda x: re.sub("__index__", "_i_", x), zz))
        for x in sorted(zz):
            print(x)

        for i,x in enumerate(useful):
            if i < 3:
                print(json.dumps(x, indent = 2))
    
        useful2 = list(filter(lambda x: get_element(x, "input/request", "") == "get", data_dict))[-1]
        pipettes = get_element(useful2, "output/data/pipettes", [])
        #print(useful2)
        #print(json_tree(useful2))

        #print(pipettes)
        pipettes2 = dict()
        for x in list(map(lambda x: {x["id"]: x["mount"]}, pipettes)):
            pipettes2.update(x) 
        print(pipettes2)

        labware = get_element(useful2, "output/data/labware", [])
        #print(labware)
        labware2 = dict()
        for x in list(map(lambda x: {x["id"]: x["location"]["slotName"]}, labware)):
            labware2.update(x) 
        print(labware2)

        #for i,x in enumerate(useful2):
        #    if i < 3:
        #        print(json.dumps(x, indent = 2))

                
        out = pd.DataFrame.from_records(list(map(extr_fn, useful)))
        out['pipetteId'] = list(map(lambda x: pipettes2[x], out['pipetteId']))
        out['labwareId'] = list(map(lambda x: labware2[x], out['labwareId']))

        print(out)
        print(out[['pipetteId', 'labwareId']])

        out.to_csv(data_json_path_to_log_asp_disp_table(p))
        #

    if EXTRACT_COMPACT_OPERATIONS in action:
        print("here we are")
        tr = convertion_functions(data_dict)
        with open(data_json_path_to_log_compact_ops(p), 'w') as f:
            for el in data_dict:
                try:
                    if is_command(el):
                        f.write(f"{command_entry_to_string(el,tr)}\n") 
                except Exception as e:
                    print("====================================")
                    print(str(e))
                    print(el)


#    if extract:
#        extract_experiment_data(p, to_extract, save_data = True, save_extraction_pathes = True)

# find all aspiration steps with mixing pipette that are dispensed at 10/A1

