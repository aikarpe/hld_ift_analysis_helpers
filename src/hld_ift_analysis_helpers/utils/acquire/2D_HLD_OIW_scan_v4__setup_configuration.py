################################################################################
# bits of useful stuff from other scripts
################################################################################

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

import sys
import functools
import json
import time
import argparse

#sys.path.append("/mnt/d/projects/HLD_parameter_determination/hld_ift_http/src") # on office pc
#sys.path.append("C:/Users/admin/Documents/Data/aikars/opentron/hld_ift_http/src") # robolab laptop
print("current contant of my python path\n: {c}".format(c = sys.path))


from hld_ift_http.opentrons_configs import Opentrons_Configuration, Instrument_Configuration, Labware_Configuration, Well_Address, Well_Configuration
from hld_ift_http.opentrons_http_comms import Opentrons_HTTP_Communications
from hld_ift_http.opentrons_pp import Opentrons_PP
from hld_ift_http.compound_properties import Compound_Properties
from hld_ift_http.solution import Solution
from hld_ift_http.solution_repository import Solution_Repository
from hld_ift_http.mixing_graph import Parent_Solution, Mixing_Vertice, Mixing_Graph
from hld_ift_http.hld_scan_1d import Scan_Graph, HLD_IFT_1D_Scan
from hld_ift_http.washing_step import Sequence_Washing_Steps
from hld_ift_http.camera_capture import Camera_Capture
from hld_ift_http.experiment_and_measurement import Experiment, Scan, Measurement, Ift_Image
from hld_ift_http.single_ift_measurement import Execute_Measurement
from hld_ift_http.autofocus import Execute_Autofocus, Execute_Autofocus_Parameters
from hld_ift_http.solution import Solution, Solution_Component
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
k = input("... press enter to continue ...")

scan_keys = params["scan"].keys()
if "oiw_scan" not in scan_keys or "oiw_scan_logout" not in scan_keys:
    print("""
        you are trying to setup OIW scan configuration!!!
        It is meant for volatile inner samples only.
        Run 2D_HLD_scan_v4__distribute_solutions_OIW.py first!!!
            (if you have done so, but see this mesage, scan_settings.json is missing
                scan/oiw_scan, and scan/oiw_scan_logout entries)
        Run this script again!!!
        """)
    k = input("...press enter to terminate...")
    exit()


# ---------- path
DATA_PATH   = params["DATA_PATH"]
LOG_PATH    = params["LOG_PATH"]
CONFIG_PATH = params["CONFIG_PATH"]
SOLUTION_REPOSITORY_PATH = params["SOLUTION_REPOSITORY_PATH"]

print(f'DEBUG: repository taken from: {SOLUTION_REPOSITORY_PATH}') #debug

# ............................................................ solution repository
rep = Solution_Repository.fromJSON(file = SOLUTION_REPOSITORY_PATH)

# ............................................................ some functions
def duplicate_config(conf_in, conf_type):
    with open(f'{CONFIG_PATH}/config_{conf_in}{conf_type}', "r") as f:
        content = f.read()
    with open(f'{CONFIG_PATH}/config_{suffix_out}{conf_type}', "w") as f:
        f.write(content)


#================================================================================
#                                                           INPUTS 
#================================================================================
#..... not_valid_out = True
#..... 
#..... logout_oiw_choice = "logout"
#..... linear_oiw_choice = "oiw"
#..... 
#..... while not_valid_out:
#.....     k = input("enter `logout` to create logout_OIW configuration, `oiw` otherwise!!! >>>")
#.....     if logout_oiw_choice in k or linear_oiw_choice in k:
#.....         not_valid_out = False
#.....         oiw_scan_status = k
#..... 
#..... logout = logout_oiw_choice in oiw_scan_status
#..... 
#..... if logout:
#.....     print("LOGOUT OIW configuration will be made!!!")
#..... else:
#.....     print("linera OIW configuration will be made!!!")

# ............................................................ config files
logout = params["scan"]["oiw_scan_logout"]

configs = params["configurations"]
suffix_in =   configs["blank_logout_OIW" if logout else "blank_OIW"]
profile_to_copy_from = configs["to_reuse"]["name"] 
suffix_out = configs["start"] 

# ............................................................ well content, locations, volumes

outer_run_sol_1       = Well_Address("9", "A1")
outer_run_sol_2       = Well_Address("9", "A2")

inner_run_sol_1       = Well_Address("2", "D1")
sample_80_20          = Well_Address("2", "D2")
sample_60_40          = Well_Address("2", "D3")
sample_40_60          = Well_Address("2", "D4")
sample_20_80          = Well_Address("2", "D5")
inner_run_sol_2       = Well_Address("2", "D6")

unused_sample_waste   = Well_Address("2", "A1")
sample_rinse_waste    = Well_Address("2", "A2")
first_rinse_waste     = Well_Address("2", "A3")
first_rinse_source    = Well_Address("2", "A4")

#================================================================================
#                              SOLUTIONS TO ASSIGN
#================================================================================
#mixtrues
#    address, name, quantity, container, list_qt, list_sol
stock_names = params["stocks"]
inputs = [
dict(address =      outer_run_sol_1, name = stock_names["outer_run_solution_1"], volume = 45000 if logout else 14500),
dict(address =      outer_run_sol_2, name = stock_names["outer_run_solution_2"], volume = 45000 if logout else 14500),

dict(address =      inner_run_sol_1, name = stock_names["inner_run_solution_1"], volume =  1400),
dict(address =         sample_80_20, name =         stock_names["sample_80_20"], volume =  1400),
dict(address =         sample_60_40, name =         stock_names["sample_60_40"], volume =  1400),
dict(address =         sample_40_60, name =         stock_names["sample_40_60"], volume =  1400),
dict(address =         sample_20_80, name =         stock_names["sample_20_80"], volume =  1400),
dict(address =      inner_run_sol_2, name = stock_names["inner_run_solution_2"], volume =  1400),

dict(address =  unused_sample_waste, name =     stock_names["washing_solution"], volume =   500), 
dict(address =   sample_rinse_waste, name =     stock_names["washing_solution"], volume =   500),
dict(address =    first_rinse_waste, name =     stock_names["washing_solution"], volume =   500), 
dict(address =   first_rinse_source, name =     stock_names["washing_solution"], volume =  1400)
]

 
wells_copy = list(map(
                    lambda x: dict(address = Well_Address.fromJSON(x)),
                    configs["to_reuse"]["content"]
                    ))

temp_op = Opentrons_PP.fromJSON(file = f'{CONFIG_PATH}/config_{profile_to_copy_from}__opentron_pp.json')

def extract_solution_info(item):
    well = temp_op.well_by_address(item["address"])
    item["solution"] = well.solution
    item["volume"] = well.volume
    item["used"] = well.used


if len(wells_copy) > 0:
    for w in wells_copy:
        extract_solution_info(w)


print(f'solrep items all: {rep.items.keys()}')

# ............................................................ modify opentron_pp object
op = Opentrons_PP.fromJSON(file = f'{CONFIG_PATH}/config_{suffix_in}__opentron_pp.json')

def update_sol_info(**kwargs):
    well = op.well_by_address(kwargs["address"])
    well.used = True
    if "solution" in kwargs.keys():
        well.solution = kwargs["solution"]
    else: 
        #debug
        print(f'update_sol_info(): name = `{kwargs["name"]}`')
        print('update_sol_info(): solutions_available ================={')
        print(rep.items.keys())
        print('update_sol_info(): solutions_available =================}')
        #debug
        well.solution = rep.items[kwargs["name"]]
    well.volume = kwargs["volume"]
    well.pipette = {}
    if (well.solution == None):
        well.volume = 0
        well.used = False


for an_input in inputs:
    update_sol_info(**an_input)
for an_input in wells_copy:
    update_sol_info(**an_input)

# ............................................................ create configurations


def m_v_concentration(solution: Solution, component: str):
   return solution.ro *  solution.components[component].w if component in solution.components.keys() else 0


if len(wells_copy) > 0:
    duplicate_config(profile_to_copy_from, "__mixing_graph.json")
else:     
    duplicate_config(suffix_in, "__mixing_graph.json")

duplicate_config(profile_to_copy_from, "__execute_measurement.json")

with open(f'{CONFIG_PATH}/config_{suffix_out}__opentron_pp.json', "w") as f:
    op.toJSON(file = f, indent = 2)

print(" ....................aaaaaaaaaaaaaaaaaaaand we are done!!!")

