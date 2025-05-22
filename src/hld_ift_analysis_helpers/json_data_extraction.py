#================================================================================
#       generic functions to extract data from json file(s)
#================================================================================

## translation of json extraction via json path
import json
import functools
import os
import re
import pandas as pd
from pandas import DataFrame
from math import nan
os.path.sep = "/"
import hld_ift_analysis_helpers.locations

def json_tree(a_dict, use_index_value: bool = False):
    """
    fn list pathes to all entries in given dictionary

    :param dict a_dict: a dictionary to explore
    :param bool use_index_value: if the flag is True a specific index is included in a path, otherwise index is excluded
    :return list: a list of strings
    :rtype: list
    """
    return json_tree_extractor(a_dict, "", use_index_value)


def json_tree_extractor(x, parent, use_index_value = False):
    """
    helper function to json_tree()

    f-n traverses dictionary tree and reports all available pathes
    """
    a_class = type(x)
    if (x is None or 
            a_class == float or 
            a_class == int or 
            a_class == bool or 
            a_class == str):
        return [parent]
    elif a_class == dict:
        keys = x.keys()
        if len(keys) < 1:
            return [parent]
        else:
            out = list(map(lambda k: json_tree_extractor(x[k], os.path.join(parent, k), use_index_value), 
                            keys
                            ))
            return list(functools.reduce(lambda x,y: x + y, out))
    else: #treat as list
        INDEX = "__index__"
        max_index = len(x)
        if max_index < 1:
            return [parent]
        if use_index_value:
            out = list(map(lambda k: json_tree_extractor(x[k], os.path.join(parent, f'{INDEX}:{k}'), use_index_value), 
                            range(max_index)
                            ))
            return list(functools.reduce(lambda x,y: x + y, out))
        else:
            out = list(map(lambda k: json_tree_extractor(x[k], os.path.join(parent, INDEX), use_index_value), 
                            range(max_index)
                            ))
            return list(functools.reduce(lambda x,y: x + y, out))

            
def extract_index(a_str):
    """
    fn extracts index value from index element in path

    index is represented by `__index__:<integer>` or `<integer>` strings
    if string parsing fails, -1 is returned

    :param str a_str: a string that is part of a path in dictionary
    :return: an integer value corresponding to given index string or -1, if parsing failes
    :rtype: int
    """
    #re_pat = re.compile("(^[[:digit:]]{1,}$)|^__index__:([[:digit:]]*)$")
    re_pat = re.compile("(^\d{1,}$)|^__index__:(\d*)$")
    a_match = re_pat.match(a_str)
    if a_match:
        #print(a_match.groups())
        return int(list(filter(lambda x: type(x) == str, a_match.groups()))[0])
    else:
        return -1


def path_split_recursive(path, a_lst = []):
    """
    fn splits a given path into all components
    """
    out = os.path.split(path)
    if out[0] == path:
        return a_lst if path == "" else [path] + a_lst
    else:
        return path_split_recursive(out[0], [out[1]] + a_lst)

#>path_split_recursive("a/b/c/d/e")
#>path_split_recursive("a\\b\\c\\d\\e")
#>path_split_recursive("D:\\a\\b\\c\\d\\e")
#>path_split_recursive("D:/a/b/c/d/e")
#>path_split_recursive("/a/b/c/d/e")

def get_element(x, path, default_value):
    """
    fn extracts an element specified by a path in a dictionary
    (to list pathes of available elements see json_tree() )

    """
    path_elements = path_split_recursive(path) 
    return element(x, path_elements, default_value)

def element(x, path, default_value):
    """
    a helper fn to get_element()
    """
    if len(path) == 0:
        return x if x is not None else default_value
    elif type(x) == dict:
        if path[0] in x.keys():
            return element(x[path[0]], path[1:], default_value)
        else:
            return default_value
    elif type(x) == list:
        index = extract_index(path[0])
        return element(x[index], path[1:], default_value) if index >= 0 and index < len(x) else default_value
    else:
        return default_value


def modified_path_factory(param_path: list, main_param_path: str):
    """
    fn creates functions to convert parameters from their general pathes
    """
    common = list(map(lambda x: os.path.commonpath([main_param_path, x]), param_path))
    common_length = list(map(lambda x: len(path_split_recursive(x)), common)) 
    
    def path_take_N_initial_bits(path, n):
        bits = path_split_recursive(path)
        return os.path.join(*[bits[x] for x in range(n)])

    fns = []
    # helper function to escape windows path separator to unix and back
    #       windows path separator messes with regex-es !!!
    def toAform(astr):
        return re.sub("\\\\", "/", astr)
    def toBform(astr):
        return re.sub("/", "\\\\", astr)

    return list(map(lambda param_p, common_p, common_len:  lambda x: toBform(re.sub(toAform(common_p), toAform(path_take_N_initial_bits(x, common_len)), toAform(param_p))),
                    param_path,
                    common,
                    common_length
                    ))

def get_elements_from_path_df(json_lst: dict, pathes_to_extract: DataFrame, default_values: list, col_names = None):
    """
    #' a helper function to extract values for given pathes
    #'
    #' @param json_lst, a list representing json dictionary with source data
    #' @param pathes_to_extract, a data.frame with pathes for each parameter to extract
    #' @param default_values, a vector of default values that is returned if path not found
    #' @returns data.frame containing extracted values
    """
    def helper_fn_factory(default_val):
        return lambda x: get_element(json_lst, x, default_val)

    data_out = None
    specified_col_names = col_names if col_names is not None and len(col_names) == len(pathes_to_extract.columns) else pathes_to_extract.columns
    for col_name_assign, col_name_current, def_val in zip(specified_col_names, pathes_to_extract.columns, default_values):
        if data_out is None:
            data_out = {col_name_assign: list(map(helper_fn_factory(def_val), pathes_to_extract[col_name_current]))}
        else:
            data_out.update({col_name_assign: list(map(helper_fn_factory(def_val), pathes_to_extract[col_name_current])) })
    return pd.DataFrame.from_dict(data_out)


def create_extraction_pahtes_df(lst_pathes, regex, generic_pathes):
    """
    fn creates DataFrame that specifies pathes of all parameters to be extracted

    tree explorations can give general path form for all parameters, 
    taking element with deepest path, pathes of other variables can be found
    """

    pat = re.compile(regex)
    main_parameter_extraction_pathes = list(filter(lambda x: pat.search(x), lst_pathes))

    generic_main_parameter_path = generic_pathes[0]
    generic_dependent_parameter_pathes = generic_pathes[1:]
    extraction_path_fn = modified_path_factory(generic_dependent_parameter_pathes, 
                                                generic_main_parameter_path)

    print(f'extraction_path_fn length: {len(extraction_path_fn)}')
    
    pathes_to_extract = pd.DataFrame(data = {generic_main_parameter_path: main_parameter_extraction_pathes}) 

    print(pathes_to_extract.head().to_string())

    for name, fn in zip(generic_dependent_parameter_pathes, extraction_path_fn):
        print(name)
        print(fn)
        a = list(map(fn, main_parameter_extraction_pathes))
        print(a[0:3])
        pathes_to_extract[name] = a
    return pathes_to_extract

#def toAform(astr):
#    return re.sub("\\\\", "/", astr)
#def toBform(astr):
#    return re.sub("/", "\\\\", astr)


def extract_experiment_data(path,
                            extraction_df,
                            save_data = True,
                            save_data_path = "",
                            save_extraction_pathes = False,
                            save_extraction_pathes_path = ""):
    # to convert "/" path sep to "\"
    extraction_df['generic_path'] = list(map(lambda x: os.path.join(*path_split_recursive(x)), extraction_df['generic_path']))

    try:
        with open(path, 'r', encoding='utf-8') as f:
            data_dict = json.load(f)
    except IOError:
        print("File not found or cannot be opened. Extraction failed!!!")
    except json.JSONDecodeError:
        print(f"Error decoding JSON file:\n{path}\n Extraction failed!!!")

    print('==== extraction df ==================')
    print(extraction_df.to_string())

    jti = json_tree(data_dict, use_index_value = True)

    pte = create_extraction_pahtes_df(jti, "ift$", extraction_df['generic_path'].tolist())
    #pte = pte.head(10)

    print('==== extraction pathes df ==================')
    print(pte.head().to_string())
    print(f'\nnumber of rows in extraction pathes df: {len(pte)}')


    if save_extraction_pathes:
        if save_extraction_pathes_path == "":
            path_temp_csv_out = os.path.join(os.path.dirname(path), "extracted_data__source.csv")
        else: 
            path_temp_csv_out = save_extraction_pathes_path
        print(f'... extraction pathes are saved at:\n{path_temp_csv_out}')
        pte.to_csv(path_temp_csv_out, index = False)

    data_extracted = get_elements_from_path_df(
                                        data_dict,
                                        pte,
                                        extraction_df['default'].tolist(),
                                        extraction_df['new_names'].tolist())

    print('==== extracted data df ==================')
    print(data_extracted.head().to_string())

    if save_data:
        if save_data_path == "":
            path_temp_extracted_csv_out = os.path.join(os.path.dirname(path), "extracted_data_.csv")
        else: 
            path_temp_extracted_csv_out = save_data_path 
        print(f'... extracted data are saved at:\n{path_temp_extracted_csv_out}')
        data_extracted.to_csv(path_temp_extracted_csv_out, index = False)


