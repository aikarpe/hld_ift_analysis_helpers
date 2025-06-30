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

# ---------- path
DATA_PATH   = params["DATA_PATH"]
LOG_PATH    = params["LOG_PATH"]
CONFIG_PATH = params["CONFIG_PATH"]
SOLUTION_REPOSITORY_PATH = params["SOLUTION_REPOSITORY_PATH"]

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

# ............................................................ config files

configs = params["configurations"]
suffix_in =   configs["blank"]
profile_to_copy_from = configs["to_reuse"]["name"] 
suffix_out = configs["start"] 

# ............................................................ well content, locations, volumes


#> slot 9   :::::::::::::::::::::::::::::::::::::::::::::::::::::: stock solutions ::::::::::::::::::::::::::::::::::      
#>         ________1________    ________2________      ________3________     ________4________      ________5________                                                                                      
#> _A_     stock_wt             stock_NaCl             stock_N810_C07        stock_N810_C16         ....                                                                                
#> _B_     ....                 ....                   ....                  ....                   ....                        
#> _C_     ....                 ....                   ....                  ....                   ....                              



stock_wt_exp               = Well_Address("9", "A1")
stock_nacl_exp             = Well_Address("9", "A2")
run_stock_surf_oil_1       = Well_Address("9", "A3")
run_stock_surf_oil_2       = Well_Address("9", "A4")
unused_sample_waste        = Well_Address("2", "A1")
sample_rinse_waste         = Well_Address("2", "A2")
first_rinse_waste          = Well_Address("2", "A3")
first_rinse_source         = Well_Address("2", "A4")

#================================================================================
#                              SOLUTIONS TO ASSIGN
#================================================================================
#mixtrues
#    address, name, quantity, container, list_qt, list_sol
stock_names = params["stocks"]
inputs = [
dict(address =         stock_wt_exp, name =      stock_names["stock_aqueous_1"], volume = 12000),
dict(address =       stock_nacl_exp, name =      stock_names["stock_aqueous_2"], volume = 12000),
dict(address = run_stock_surf_oil_1, name = stock_names["run_stock_surf_oil_1"], volume = 12000),
dict(address = run_stock_surf_oil_2, name = stock_names["run_stock_surf_oil_2"], volume = 12000),
dict(address =  unused_sample_waste, name =                             "water", volume =   500), 
dict(address =   sample_rinse_waste, name =                             "water", volume =   500),
dict(address =    first_rinse_waste, name =                             "water", volume =   500), 
dict(address =   first_rinse_source, name =                             "water", volume =  1400)
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

