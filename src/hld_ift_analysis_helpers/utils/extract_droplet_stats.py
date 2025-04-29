import sys 
sys.path.append("D:/projects/HLD_parameter_determination/hld_ift_analysis_helpers/src")

#from scripts.json_data_extraction import *
from hld_ift_analysis_helpers.montage_bits import *
from hld_ift_analysis_helpers.droplet_stats import *
from hld_ift_analysis_helpers.collect_files_folders import match_conc_at_end, collect_dirs

#import pandas as pd
#================================================================================
# this part generates all output from all available droplets!!! takes a very long time

raw_data_folder = r'\\huckdfs-srv.science.ru.nl\huckdfs\RobotLab\Storage-Miscellaneous\aigars\temp\AOT_IB-45_C7C16_NaCl'
#2>folders_use = match_conc_at_end(collect_dirs(raw_data_folder))

folders_use = list(filter(lambda x: re.search("exp_2025-04", x), 
                            match_conc_at_end(collect_dirs(raw_data_folder))))
#--2-->folders_use = list(filter(lambda x: re.search("exp_2025-03-2", x), 
#--2-->                            match_conc_at_end(collect_dirs(raw_data_folder))))
#--1-->folders_use = list(filter(lambda x: re.search("03-24", x), 
#--1-->                            match_conc_at_end(collect_dirs(raw_data_folder))))

#quick test |> folders_use = [folders_use[0]]

first = True
path_out = os.path.join(raw_data_folder, "droplet_stats_data_2025_04_29.output.csv") #"data.output_temp.csv")
if os.path.exists(path_out):
    os.remove(path_out)
for p in folders_use:
    print(p)
    process_dir(p, path_out)
    
#================================================================================




