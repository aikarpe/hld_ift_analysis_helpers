################################################################################
#   A single measurement of solution from address `2/D6` (1000uL)
#       - can run using existing setting files,
#       - experiment name defaults to `exp_{timestamp}`
# 
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
#sys.path.append("C:/Users/admin/Documents/Data/aikars/opentron/hld_ift_http/src") # robolab laptop
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

#index = str(params["scan"]["scan_part_index"])
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

#print("\n\n\n")
#print("================================================================================")
#print("================================================================================")
#print("================================================================================")
#print(f'Following wells will be used in this scan: ')
#print(f'                                           {wells_info[index]["info"]}!!!')
#print("")
#print("make sure they are open and 1400 mkL of oil is added there")
#print("")
#print("================================================================================")
#print("================================================================================")
#print("================================================================================")

#k = input("... press enter to continue ...")

# ---------- path
DATA_PATH   = params["DATA_PATH"]
LOG_PATH    = params["LOG_PATH"]
CONFIG_PATH = params["CONFIG_PATH"]
SOLUTION_REPOSITORY_PATH = params["SOLUTION_REPOSITORY_PATH"]

# 1st run
configs = params["configurations"]
suffix_in =  configs["start"]
suffix_out = configs["end"] #+ index


#> slot 9   :::::::::::::::::::::::::::::::::::::::::::::::::::::: stock solutions ::::::::::::::::::::::::::::::::::      
#>         ________1________    ________2________      ________3________     ________4________      ________5________                                                                                      
#> _A_     stock_wt             stock_NaCl             stock_N810_C07        stock_N810_C16         ....                                                                                
#> _B_     ....                 ....                   ....                  ....                   ....                        
#> _C_     ....                 ....                   ....                  ....                   ....                              

#stock_1_loc = wells_info[index]["1"]
#stock_2_loc = wells_info[index]["2"]
#stock_wt   = Well_Address("9", "A3") 
#stock_NaCl = Well_Address("9", "A4")
stock_1_loc = Well_Address("2", "A1")
stock_2_loc = Well_Address("2", "A2")
stock_wt   = Well_Address("2", "A1") 
stock_NaCl = Well_Address("2", "A2")

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
        "", #force to use date_and_time string as experiment name
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

with open(f'{CONFIG_PATH}/config_{suffix_in}__execute_measurement.json', "r") as file:
    meas_config = json.load(file)

k = input("exclude washing steps from measurements ([y]es/[n]o):>>")

if len(k) > 0 and k[0] == 'y':
    meas_config["washing_steps"]["sequence_washing_steps"] = []

print("===============================meas_config")
print(meas_config)
print("==========================================")

ift_measurement = Execute_Measurement.fromJSON(
            cfg = meas_config,
            opentron = op,
            camera = camera,
            experiment = exp
            )

print(f'{CONFIG_PATH}/config_{suffix_in}__execute_measurement.json')
k = input("press ENTER..........")

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
#print_list_of_wells("all wells", op.wells())
#print_list_of_wells("wells to allocate in scan", to_use)

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

# add water to measurement location
#source_location = Well_Address("9", "A1")
source_location = Well_Address("2", "A1") # all configurations should have a washing solution at this location, nature of solution is not importatn as long as it is defined!!!
water = op.well_by_address(source_location)

meas_loc = Well_Address("10","A1")
#well = op.well_by_address(meas_loc)
#well.used = True
#well.solution = water.solution
#well.volume = 3000
#well.pipette = {}
#if (well.solution == None):
#    well.volume = 0
#    well.used = False



exp.new_scan(active = True)
#[expand scan_graph n_expansions times ]
#self.scan_graph.expand(self.n_expansions)
a_pipette = "left"
mixing_pipette = "right"

loc_to_measure = Well_Address("2", "D6")

well_to_test = op.well_by_address(loc_to_measure)
well_to_test.used = True
well_to_test.solution = water.solution
well_to_test.volume = 1000
well_to_test.pipette = {}
if (well_to_test.solution == None):
    well_to_test.volume = 0
    well_to_test.used = False


intent = "setup"

w1 = op.well_by_address(loc_to_measure)
print("<test_point>========================================")
print(f'@address: {loc_to_measure.toDict()}\n content: {w1.solution.toDict()}')
k = input("...enter...")

op.clear_run_if_needed()
op.create_run() # .... starting setup a new http script run
op.home()

op.pick_up_tip_safely_for(
                          a_pipette,
                          Well_Address("10", "A1"),
                          intent) 



k = input("adjust needle, press Enter ...")

k = input("how many measurements to perform? >>>")

try:
    val = int(float(k)) 
    n = 1 if val < 1 else 9 if val > 9 else val
except:
    n = 1

for i in range(n):
    if i != 0:
        op.next_cuvette(mixing_pipette, well.address.slot)

    well = op.well_by_address(meas_loc)
    well.used = True
    well.solution = water.solution
    well.volume = 3000
    well.pipette = {}
    if (well.solution == None):
        well.volume = 0
        well.used = False

    print(f'@address: {meas_loc.toDict()}\n content: {op.well_by_address(meas_loc).solution.toDict()}')
   
    ift_measurement.measure(loc_to_measure, f"conc_{i/n:.05f}", "")


op.drop_tip_at_origin(a_pipette, intent)



#hld_scan = HLD_IFT_2D_Scan_W_Slider(
#                opentron = op,
#                n_expansions = hld_scan_args["n_expansions"],
#                n_approximation = hld_scan_args["n_approximation"],
#                scan_graph = scan,
#                mixing_pipette = MIXING_PIPETTE,
#                experiment = exp,
#                measurement = ift_measurement,
#                number_of_oil_points = hld_scan_args["number_of_oil_points"],
#                oil_volume = hld_scan_args["oil_volume"],
#                oil_1_address = stock_NaCl,
#                oil_2_address = stock_wt,
#                scan_type = hld_scan_args["scan_type"],
#                restart_index = hld_scan_args["restart_index"],
#                manual_open_close_sample_reservoir = False,
#                scan_params = {}
#                )

record_all_configs(f'{suffix_out}_start')

#hld_scan.run()
    
record_all_configs(f'{suffix_out}_end')
exp.saveConfig()


print(" ....................aaaaaaaaaaaaaaaaaaaand we are done!!!")
