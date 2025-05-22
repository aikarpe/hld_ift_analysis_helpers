import os
import functools

def exp_root_fldr_to_processed_fldr_path(a_path):
    return os.path.join(a_path, "processed")

def data_json_path_to_processed_data_json_path(a_path):
    fldr, fnm = os.path.split(a_path)
    if fnm == "data.json":
        processed_fldr_path = exp_root_fldr_to_processed_fldr_path(fldr) 
        os.makedirs(processed_fldr_path, exist_ok = True)
        return os.path.join(processed_fldr_path, "data_processed.json")
    else:
        raise Exception("Not a data.json file path")

def data_json_path_to_exp_root_path(a_path):
    fldr, fnm = os.path.split(a_path)
    if fnm == "data.json":
        return fldr
    else:
        raise Exception("Not a data.json file path")


def drop_stats_path(a_path):
    fldr, fnm = os.path.split(a_path)
    if fnm == "data.json":
        processed_fldr_path = exp_root_fldr_to_processed_fldr_path(fldr) 
        os.makedirs(processed_fldr_path, exist_ok = True)
        return os.path.join(processed_fldr_path, "drop_stats.csv")
    else:
        raise Exception("Not a data.json file path")

def extracted_data_path(a_path):
    fldr, fnm = os.path.split(a_path)
    if fnm == "data.json":
        processed_fldr_path = exp_root_fldr_to_processed_fldr_path(fldr) 
        os.makedirs(processed_fldr_path, exist_ok = True)
        return os.path.join(processed_fldr_path, "extracted_data_.csv")
    else:
        raise Exception("Not a data.json file path")


def extracted_data_source_path(a_path):
    fldr, fnm = os.path.split(a_path)
    if fnm == "data.json":
        processed_fldr_path = exp_root_fldr_to_processed_fldr_path(fldr) 
        os.makedirs(processed_fldr_path, exist_ok = True)
        return os.path.join(processed_fldr_path, "extracted_data__source.csv")
    else:
        raise Exception("Not a data.json file path")


def is_data_json(a_path):
    return os.path.split(a_path)[1] == "data.json"

def data_json_path_to_processed_item_path(a_path, cmpnnts):
    if is_data_json(a_path):
        root, nm = os.path.split(a_path)
        if type(cmpnnts) == str:
            return os.path.join(root, cmpnnts)
        else:
            return functools.reduce(lambda x, y: os.path.join(x, y), cmpnnts, root)
    else:
        raise Exception("Not a data.json file path")

def data_json_path_to_exp_montage_path(a_path):
    return data_json_path_to_processed_item_path(a_path, ["processed", "montages", "exp_montage.jpg"])
def data_json_path_to_measurement_montage_fldr_path(a_path):
    return data_json_path_to_processed_item_path(a_path, ["processed", "montages", "measurement"])
def data_json_path_to_log_compact_ops(a_path): 
    return data_json_path_to_processed_item_path(a_path, ["processed", "log_digest", "compact_ops_lst.txt"])
def data_json_path_to_log_keys(a_path): 
    return data_json_path_to_processed_item_path(a_path, ["processed", "log_digest", "unique_json_keys.txt"])
def data_json_path_to_log_urls(a_path): 
    return data_json_path_to_processed_item_path(a_path, ["processed", "log_digest", "command_urls.txt"])
def data_json_path_to_log_asp_disp_table(a_path): 
    return data_json_path_to_processed_item_path(a_path, ["processed", "log_digest", "aspiration_dispensation_params.csv"])
def data_json_path_to_raw_log(a_path): 
    if is_data_json(a_path):
        root, nm = os.path.split(a_path)
        return os.path.join(root, "log.log")
    else:
        raise Exception("Not a data.json file path")
