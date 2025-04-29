# script provides convenient way to create or edit and saves dictionary of extraction options as
#       data.json_extraction_options.json file

import json
from math import nan as nan
import os
DEFAULT_FILE_NAME = 'data.json_extraction_options.json'

folder_out = r'\\huckdfs-srv.science.ru.nl\huckdfs\RobotLab\Storage-Miscellaneous\aigars\temp\AOT_IB-45_C7C16_NaCl'

if not os.path.isdir(folder_out):
    print(f'folder_out:\n   {folder_out}\nis not a folder, no options saved!\n ... quiting')

path_out = os.path.join(folder_out, DEFAULT_FILE_NAME)

out = {

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
#       "11": [0,           "scans/_i_/measurements/_i_/solution_outer/components/BrijL4/w",        "so_Brij_L4_w"],                                                       
       "11": [0,           "scans/_i_/measurements/_i_/solution_outer/components/aot/w",            "so_AOT_w"],                                                       
       "12": [0,           "scans/_i_/measurements/_i_/solution_outer/components/hexadecane/w",     "so_hexadecane_w"],                                                    
       "13": [0,           "scans/_i_/measurements/_i_/solution_outer/components/heptane/w",        "so_heptane_w"],                                                
       "14": [nan,         "scans/_i_/measurements/_i_/delta_ro",                                   "delta_ro"]
            }

with open(path_out, "w") as f:
    print(f' saving data.json extraction options in:\n   {path_out}')
    json.dump(out, f, indent = 2)

