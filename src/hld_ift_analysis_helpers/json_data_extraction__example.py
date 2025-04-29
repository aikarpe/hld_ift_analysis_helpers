import sys 
sys.path.append("D:/projects/HLD_parameter_determination/hld_ift_analysis_helpers/scripts")

#from scripts.json_data_extraction import *
from json_data_extraction import *
import pandas as pd

to_extract = pd.DataFrame.from_dict({

        "0": [nan,         "scans/_i_/measurements/_i_/ift_images/_i_/ift",                         "ift"],                                                                
        "1": ["__NA__" ,   "scans/_i_/measurements/_i_/measurement_folder",                         "measurement_folder"],                                                 
        "2": ["__NA__" ,   "scans/_i_/measurements/_i_/label",                                      "label"],                                                              
        "3": [nan,         "scans/_i_/measurements/_i_/needle_dia",                                 "needle_dia"],                                                         
        "4": ["__NA__" ,   "scans/_i_/measurements/_i_/ift_images/_i_/path",                        "path"],                                                               
        "5": [nan,         "scans/_i_/measurements/_i_/y",                                          "y"],                                                                  
        "6": [nan,         "scans/_i_/measurements/_i_/concentration",                              "concentration"],                                                      
        "7": [nan,         "scans/_i_/measurements/_i_/solution_inner/ro",                          "si_ro"],                                                              
        "8": [0,           "scans/_i_/measurements/_i_/solution_inner/components/water/w",          "si_water_w"],                                                         
        "9": [0,           "scans/_i_/measurements/_i_/solution_inner/components/NaCl/w",           "si_NaCl_w"],                                                           
       "10": [nan,         "scans/_i_/measurements/_i_/solution_outer/ro",                          "so_ro"],                                                              
       "11": [0,           "scans/_i_/measurements/_i_/solution_outer/components/Brij_L4/w",        "so_Brij_L4_w"],                                                       
       "12": [0,           "scans/_i_/measurements/_i_/solution_outer/components/hexadecane/w",     "so_hexadecane_w"],                                                    
       "13": [0,           "scans/_i_/measurements/_i_/solution_outer/components/heptane/w",        "so_heptane_w"],                                                
       "14": [nan,         "scans/_i_/measurements/_i_/delta_ro",                                   "delta_ro"]
            },
            orient = "index",
            columns = ['default', 'generic_path', 'new_names'])


file_path = [
            "D:/temp_data/exp_2025-02-07_001_2.5g_BrijL4_C7-C16/data.json",
            "D:/temp_data/exp_2025-02-13_001_5g_BrijL4_C7-C16/data.json",
            # "D:/temp_data/exp_2025-02-14_001_10g_BrijL4_C7-C16/data.json",
            "D:/temp_data/exp_2025-02-24_001_20g_BrijL4_C7-C16/data.json"
            ]

for p in file_path:
    extract_experiment_data(p, to_extract, save_data = True, save_extraction_pathes = True)


