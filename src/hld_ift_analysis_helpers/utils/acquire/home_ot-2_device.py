################################################################################
#   2D scan with manual cuvette substitution
#       A fixed number of oil mixtures that are pipetted into cuvette,
#           but manually supervised
################################################################################
#   
#   Set cuvette volume to 0 and solution to None
#   Pause action (to place empty cuvette in place)
#   mix oil and pipet into cuvette
#   Pause action (to close oil reservoirs)
#   perform HLD scan with this oil

#=======================================================================================================
# slot 9
#=======================================================================================================
#       1                       2                   3               4                   5
#
# A:    .                       .                   stock_water     stock_NaCl          .                 
#                                                   14500           14500                         
#
# B:    .                       .                   .               .                   .                                                                               
#                                                                                                                                                                       
#
# C:    .                       .                   .               .                   .
#                                                                                       
#=======================================================================================================

#=======================================================================================================
# slot 2
#=======================================================================================================
#       1           2           3           4               5           6
#
# A:    hexadecane  hexadecane  hexadecane  <sample_mix>    .           .
#       500         500         500         1400           
#
# B:    .           .           .           .               .           .  
#      
#
# C:    run_stock_  sample_     sample_     sample_         sample_     run_stock_
#       surf_oil_1  80_20       60_40       40_60           20_80       surf_oil_2  
#       1400        1400        1400        1400            1400        1400
#=======================================================================================================


# ================================================================================

# prep scan

################################################################################
# bits of useful stuff from other scripts
################################################################################
import sys
import functools
import json
import time
import argparse

#sys.path.append("/mnt/d/projects/HLD_parameter_determination/hld_ift_http/src") # on office pc
sys.path.append("C:/Users/admin/Documents/Data/aikars/opentron/hld_ift_http/src") # robolab laptop
print("current contant of my python path\n: {c}".format(c = sys.path))


from hld_ift_http.opentrons_configs import Opentrons_Configuration, Instrument_Configuration, Labware_Configuration, Well_Address, Well_Configuration
from hld_ift_http.opentrons_http_comms import Opentrons_HTTP_Communications
from hld_ift_http.opentrons_pp import Opentrons_PP
from hld_ift_http.compound_properties import Compound_Properties
from hld_ift_http.solution import Solution
from hld_ift_http.mixing_graph import Mixing_Graph
from hld_ift_http.hld_scan_1d import Scan_Graph, HLD_IFT_2D_Scan_W_Slider
from hld_ift_http.washing_step import Sequence_Washing_Steps
from hld_ift_http.camera_capture import Camera_Capture
from hld_ift_http.experiment_and_measurement import Experiment, Scan, Measurement, Ift_Image
from hld_ift_http.single_ift_measurement import Execute_Measurement
from hld_ift_http.autofocus import Execute_Autofocus, Execute_Autofocus_Parameters
import hld_ift_http.errors

parser = argparse.ArgumentParser()
parser.add_argument("source", help = "source of 2D oil scan configuration")
args = parser.parse_args()

try:
    with open(args.source, "r") as f:
        params = json.load(f)
except Exception as e:
    print(str(e))
    exit()

print(json.dumps(params))

index = str(params["scan"]["scan_part_index"])
wells_info  = {
                "1": {
                    "info": "2/D1, 2/D2", 
                    "1": Well_Address("2", "D1"),
                    "2": Well_Address("2", "D2")
                     },
                "2": {
                    "info": "2/D3, 2/D4",
                    "1": Well_Address("2", "D3"),
                    "2": Well_Address("2", "D4")
                     },
                "3": {
                    "info": "2/D5, 2/D6", 
                    "1": Well_Address("2", "D5"),
                    "2": Well_Address("2", "D6")
                     }
              }

print("\n\n\n")
print("================================================================================")
print("================================================================================")
print("================================================================================")
print(f'Following wells will be used in this scan: ')
print(f'                                           {wells_info[index]["info"]}!!!')
print("")
print("make sure they are open and 1400 mkL of oil is added there")
print("")
print("================================================================================")
print("================================================================================")
print("================================================================================")

k = input("... press enter to continue ...")

# ---------- path
DATA_PATH   = params["DATA_PATH"]
LOG_PATH    = params["LOG_PATH"]
CONFIG_PATH = params["CONFIG_PATH"]
SOLUTION_REPOSITORY_PATH = params["SOLUTION_REPOSITORY_PATH"]

# 1st run
configs = params["configurations"]
suffix_in =  configs["start"]
suffix_out = configs["end"] + index


#> slot 9   :::::::::::::::::::::::::::::::::::::::::::::::::::::: stock solutions ::::::::::::::::::::::::::::::::::      
#>         ________1________    ________2________      ________3________     ________4________      ________5________                                                                                      
#> _A_     stock_wt             stock_NaCl             stock_N810_C07        stock_N810_C16         ....                                                                                
#> _B_     ....                 ....                   ....                  ....                   ....                        
#> _C_     ....                 ....                   ....                  ....                   ....                              

stock_1_loc = wells_info[index]["1"]
stock_2_loc = wells_info[index]["2"]
stock_wt   = Well_Address("9", "A3") 
stock_NaCl = Well_Address("9", "A4")

oil_points = 6
oil_volume = 3000

MIXING_PIPETTE = "right"

# ---------- initialization of 
# ............................................................ opentron_pp object

op = Opentrons_PP.fromJSON(file = f'{CONFIG_PATH}/config_{suffix_in}__opentron_pp.json', log_path = LOG_PATH)

op.clear_run_if_needed()
op.create_run() # .... starting setup a new http script run
op.home()


print(" ....................aaaaaaaaaaaaaaaaaaaand we are done!!!")
