################################################################################
# script for Adam to visualize flow and rest phases during hld_ift measurement
#   for a method paper to imporve figures
################################################################################
#
# typical measurement has:
#       20 steps,
#       each step dispenses 0.05 mkL of liquid at 0.013 mkL/s rate
#       push phase is `0.05 / 0.013 = 3.846` s long
#       total duration of push phase is  `0.05 / 0.013 * 20 = 76.9` s
#       rest phase is 6 s long
#       total length of push/rest cycles should be
#            `0.05 / 0.013 * 20 + 19 * 6 = 190.9` s
#
# example:
# python C:\Users\agaiosa\code\hld_ift_analysis_helpers\src\hld_ift_analysis_helpers\utils\method_paper\copy_flow_and_rest_phase_images.py -n=20 -d=4.0 -f=3.846 -p=10.196 -v "//huckdfs-srv.science.ru.nl/huckdfs/robotlab/Storage-Miscellaneous/aigars/temp/HLD_scan/Ecosurf_EH-3_OIW_v2/exp_2025-12-16_20g_Ecosurf_EH-3_C7_C16_OIW_0012" "C:/Users/agaiosa/Downloads/adam_copy/test" 
#

import functools
import shutil
import os
from hld_ift_analysis_helpers.collect_files_folders import collect_dirs, match_conc_at_end
import argparse


# ==== index selection =======================================
def within_boundaries(x, boundaries):
    for start,end in boundaries:
        if x < start:
            return False
        if x < end:
            return True
    return False

# === index to image name =====================================
def pretty_index(i):
    return f'{i:05d}'

def image_name(i):
    return f'{pretty_index(i)}.jpg'

# === copy ====================================================
def copy_folder(source_path, target_path, index):
    for x in index:
        source_file = os.path.join(source_path, image_name(x))
        if os.path.exists(source_file): 
            shutil.copy(source_file, target_path)  
    
def copy_state_images(root, fldr, source_folder, index):
    temp, conc = os.path.split(source_folder)
    _ ,scan = os.path.split(temp)

    target_fldr = os.path.join(root, fldr, scan, conc)

    if not os.path.exists(target_fldr):
        os.makedirs(target_fldr)

    print(f'{source_folder} ===> {target_fldr}')
    copy_folder(source_folder, target_fldr, index)
        

# index of flow and rest phase

parser = argparse.ArgumentParser()
parser.add_argument("source", help = "path to folder with experiment to process")
parser.add_argument("target", help = "path to folder where processed data will be stored")
parser.add_argument("-d", "--delay", type = float, help = "delay in seconds to beginning of first flow phase", default = 0.0)
parser.add_argument("-f", "--flow_phase_duration", type = float, help = "duration of a flow phase in seconds", default = 3.846)
parser.add_argument("-p", "--period_duration", type = float, help = "a duration of flow and rest phase in seconds", default = 9.846)
parser.add_argument("-n", "--number_of_steps", type = int, help = "number of flow phases", default = 20)
parser.add_argument("-v", "--view", help = "view index selection, without copying", action = "store_true")

args = parser.parse_args()

delta = args.delay #4.0
duration = args.flow_phase_duration #3.846 # 0.05 / 0.0133333
period = args.period_duration #duration + 6.35
n_periods = args.number_of_steps #20

boundaries = [(delta + i * period, delta + i * period + duration) for i in range(n_periods)]

boundaries

flow_index =       [x for x in range(200) if within_boundaries(float(x), boundaries)]
rest_phase_index = [x for x in range(200) if not within_boundaries(float(x), boundaries) and float(x) > delta ]

source_root = args.source 
target_root = args.target
#'//huckdfs-srv.science.ru.nl/huckdfs/robotlab/Storage-Miscellaneous/aigars/temp/HLD_scan/Ecosurf_EH-3_OIW_v2/exp_2025-12-16_20g_Ecosurf_EH-3_C7_C16_OIW_0012' 'C:/Users/agaiosa/Downloads/adam_copy/test' -n=20 -d=4.0 -f=3.846 -p=10.196

if not args.view:
    source_folders = match_conc_at_end(collect_dirs(source_root))

    for a_type,an_index in zip(["flow", "rest"], [flow_index, rest_phase_index]):
        for a_source in source_folders:
            copy_state_images(target_root, a_type, a_source, an_index)

else:
    print(f'flow index:\n{flow_index}\nrest phase index:\n{rest_phase_index}')


