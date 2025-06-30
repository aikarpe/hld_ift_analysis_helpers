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
k = input("... press enter to continue ...")

# ---------- path
DATA_PATH   = params["DATA_PATH"]
LOG_PATH    = params["LOG_PATH"]
CONFIG_PATH = params["CONFIG_PATH"]
SOLUTION_REPOSITORY_PATH = params["SOLUTION_REPOSITORY_PATH"]

# 1st run
configs = params["configurations"]
suffix_in =  configs["start"]
suffix_out = configs["end"]


#> slot 9   :::::::::::::::::::::::::::::::::::::::::::::::::::::: stock solutions ::::::::::::::::::::::::::::::::::      
#>         ________1________    ________2________      ________3________     ________4________      ________5________                                                                                      
#> _A_     stock_wt             stock_NaCl             stock_N810_C07        stock_N810_C16         ....                                                                                
#> _B_     ....                 ....                   ....                  ....                   ....                        
#> _C_     ....                 ....                   ....                  ....                   ....                              

stock_1_loc = Well_Address("9", "A1")
stock_2_loc = Well_Address("9", "A2")
heptane_well_address = Well_Address("9", "A3") 
hexadecane_well_address = Well_Address("9", "A4")

oil_points = 6
oil_volume = 3000

MIXING_PIPETTE = "right"

# ---------- initialization of 
# ............................................................ Camera_Capture object
camera = Camera_Capture()

# ............................................................ Experiment object (data capture)
exp_metadata = params["scan"]["experiment_metadata"]
exp = Experiment(
        DATA_PATH,
        f"exp_{suffix_out}",
        description = exp_metadata["description"],
        needle_dia = exp_metadata["needle_dia"],
        oil = exp_metadata["oil"],
        measurement = exp_metadata["measurement"],
        scan_type = exp_metadata["scan_type"],
        suffix = suffix_out
        )

# ............................................................ opentron_pp object

op = Opentrons_PP.fromJSON(file = f'{CONFIG_PATH}/config_{suffix_in}__opentron_pp.json', log_path = exp.log_path())


# ............................................................ Mixing_Graph object
#                                                       general graph to store all
#                                                       mixing dependencies during
#                                                       this run
mixing_graph = Mixing_Graph.fromJSON(file = f'{CONFIG_PATH}/config_{suffix_in}__mixing_graph.json')


# ............................................................ Execute_Measurement object
ift_measurement = Execute_Measurement.fromJSON(
            file = f'{CONFIG_PATH}/config_{suffix_in}__execute_measurement.json',
            opentron = op,
            camera = camera,
            experiment = exp
            )


### test heigth vs volume calculation
#wash_loc = Well_Address("2", "A1")
#wash_well = op.well_by_address(wash_loc)
#print(f'TEST: \naddress: {wash_loc}\nz: {op.z_at_volume(wash_loc)}\nvolume: {op.well_by_address(wash_loc).volume}')
#for v in range(22):
#    vol = v / 21 * 1500
#    wash_well.set_volume(vol) 
#    print(f'volumes: {op.z_at_volume(wash_loc):4.0f}   {wash_well.volume: 4.0f}')
#exit()

# ............................................................ Wells to use
def print_list_of_wells(label, lst):
    print(f"############################# {label}:")
    for w in lst:
        print(w.toDict()) 
    print(f"############################# END OF {label}")

print(f'op is objecte is {type(op)}')

to_use = list(filter(lambda x: not x.used and \
                                x.geometry_label == "vial_1500ul" and \
                                x.available,
                                op.wells()))
to_use = list(map(lambda x: x.address, to_use))
print_list_of_wells("all wells", op.wells())
print_list_of_wells("wells to allocate in scan", to_use)

#print(f'type of 1st well is {type(to_use[0])}')
#k = input("press enter to cont")
# ............................................................ slot "9": stocks

scan = Scan_Graph(
            a_mixing_graph = mixing_graph,
            well_1 = stock_1_loc,
            well_2 = stock_2_loc,
            scan_label = f"scan_{suffix_out}",
            wells_to_use = to_use
            )

# ............................................................ config saving bits
def record_all_configs(suffix):
    with open(f'{CONFIG_PATH}/config_{suffix}__opentron_pp.json', "w") as f:
        op.toJSON(file = f, indent = 2)
    
    with open(f'{CONFIG_PATH}/config_{suffix}__mixing_graph.json', "w") as f:
        mixing_graph.toJSON(file = f, indent = 2)
    
    with open(f'{CONFIG_PATH}/config_{suffix}__execute_measurement.json', "w") as f:
        ift_measurement.toJSON(file = f, indent = 2)
    

# ............................................................ HLD_IFT_1D_Scan object


# ............................................................ oil scan parameters

hld_scan_args = params["scan"]
hld_scan = HLD_IFT_2D_Scan_W_Slider(
                opentron = op,
                n_expansions = hld_scan_args["n_expansions"],
                n_approximation = hld_scan_args["n_approximation"],
                scan_graph = scan,
                mixing_pipette = MIXING_PIPETTE,
                experiment = exp,
                measurement = ift_measurement,
                number_of_oil_points = hld_scan_args["number_of_oil_points"],
                oil_volume = hld_scan_args["oil_volume"],
                oil_1_address = hexadecane_well_address,
                oil_2_address = heptane_well_address,
                scan_type = hld_scan_args["scan_type"],
                scan_params = {}
                )

record_all_configs(f'{suffix_out}_start')

hld_scan.run()
    
record_all_configs(suffix_out)
exp.saveConfig()


print(" ....................aaaaaaaaaaaaaaaaaaaand we are done!!!")
