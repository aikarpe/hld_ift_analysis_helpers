################################################################################
# a script to create run stocks for scan with water soluble and oil soluble surfactants
#   Script assumes that:
#       one surfactant is in oil; another surfactant is in aqueous phase
#       each surfactant has two main stocks and each main stock has associated dilution liquid
#src/hld_ift_analysis_helpers/utils/acquire/pick_up_pipette_tips_bunch.py
################################################################################
import sys
import functools
import json
import time
import argparse

#sys.path.append("/mnt/d/projects/HLD_parameter_determination/hld_ift_http/src") # on office pc
#> sys.path.append("C:/Users/admin/Documents/Data/aikars/opentron/hld_ift_http/src") # robolab laptop
print("current contant of my python path\n: {c}".format(c = sys.path))


from hld_ift_http.opentrons_configs import Opentrons_Configuration, Instrument_Configuration, Labware_Configuration, Well_Address, Well_Configuration
from hld_ift_http.opentrons_http_comms import Opentrons_HTTP_Communications
from hld_ift_http.opentrons_pp import Opentrons_PP
from hld_ift_http.compound_properties import Compound_Properties
from hld_ift_http.solution import Solution
from hld_ift_http.mixing_graph import Parent_Solution, Mixing_Vertice, Mixing_Graph
from hld_ift_http.solution_repository import Solution_Repository
from hld_ift_http.hld_scan_1d import Scan_Graph, HLD_IFT_1D_Scan
from hld_ift_http.washing_step import Sequence_Washing_Steps
from hld_ift_http.camera_capture import Camera_Capture
from hld_ift_http.experiment_and_measurement import Experiment, Scan, Measurement, Ift_Image
from hld_ift_http.single_ift_measurement import Execute_Measurement
from hld_ift_http.autofocus import Execute_Autofocus, Execute_Autofocus_Parameters
from hld_ift_http.solution import Solution, Solution_Component
from hld_ift_http.opentron_well_status import pretty_config_make, solution_name_str, well_address_str 
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

# ............................................................ inputs
suffix_in = params["configurations"]["blank"] #"hld_ift_experiment__blank"
suffix_out = "something"

pipette = "right"

# ............................................................ modify opentron_pp object
op = Opentrons_PP.fromJSON(file = f'{CONFIG_PATH}/config_{suffix_in}__opentron_pp.json')


# ---------- record initial configurations 

def duplicate_config(conf_type):
    with open(f'{CONFIG_PATH}/config_{suffix_in}{conf_type}', "r") as f:
        content = f.read()
    with open(f'{CONFIG_PATH}/config_{suffix_out}{conf_type}', "w") as f:
        f.write(content)


c_surf_out_stock = params["c_surfactant_outer_stock"]
c_surf_out_exp   = params["c_surfactant_outer_run_solution"]
v_stock_out      = c_surf_out_exp / c_surf_out_stock
v_diluter_out    = (c_surf_out_stock - c_surf_out_exp) / c_surf_out_stock
print(f'c_surf_out_stock: {c_surf_out_stock}')
print(f'c_surf_out_exp: {c_surf_out_exp}')
print(f'v_stock_out: {v_stock_out}')
print(f'v_diluter_out: {v_diluter_out}')


c_surf_in_stock =  params["c_surfactant_inner_stock"]
c_surf_in_exp   =  params["c_surfactant_inner_run_solution"]
print(f'c_surf_in_exp: {c_surf_in_exp}')
print(f'c_surf_in_stock: {c_surf_in_stock}')

v_stock_in     =  c_surf_in_exp / c_surf_in_stock 
v_diluter_in   = 1 - v_stock_in

stock_out_1    = Well_Address("9", "B1") #"A1")
stock_out_2    = Well_Address("9", "B2") #"A2")
diluter_out_1  = Well_Address("9", "C1") #"A3")
diluter_out_2  = Well_Address("9", "C2") #"A4")
stock_wash     = Well_Address("9", "C5") #"A5")  
stock_in_1     = Well_Address("9", "B3")
stock_in_2     = Well_Address("9", "B4")
diluter_in_1   = Well_Address("9", "C3") #"B3")
diluter_in_2   = Well_Address("9", "C4") #"B4")

#================================================================================
#                              STOCK DETAILS 
#================================================================================
#inputs
#    address, soluton, quantity, container
stock_names = params["stocks"]
stocks = [
dict(address =   stock_out_1, solution =    stock_names["outer_stock_1"], volume = 12000), 
dict(address =   stock_out_2, solution =    stock_names["outer_stock_2"], volume = 12000), 
dict(address = diluter_out_1, solution =  stock_names["outer_diluter_1"], volume = 12000), 
dict(address = diluter_out_2, solution =  stock_names["outer_diluter_2"], volume = 12000),
dict(address =    stock_wash, solution = stock_names["washing_solution"], volume = 12000),
dict(address =    stock_in_1, solution =    stock_names["inner_stock_1"], volume = 12000), 
dict(address =    stock_in_2, solution =    stock_names["inner_stock_2"], volume = 12000), 
dict(address =  diluter_in_1, solution =  stock_names["inner_diluter_1"], volume = 12000), 
dict(address =  diluter_in_2, solution =  stock_names["inner_diluter_2"], volume = 12000) 
]


#find solution in repository
#from opentronpp get well with given address
#updates well with solution, volume

outer_run_sol_1     = Well_Address("9", "A1") #"C1")
outer_run_sol_2     = Well_Address("9", "A2") #"C2")
inner_run_sol_1     = Well_Address("9", "A3") #"C3")
inner_run_sol_2     = Well_Address("9", "A4") #"C4")
unused_sample_waste = Well_Address("2", "A1")
sample_rinse_waste  = Well_Address("2", "A2")
first_rinse_waste   = Well_Address("2", "A3")
first_rinse_source  = Well_Address("2", "A4")

#================================================================================
#                              SOLUTIONS NEEDED
#================================================================================
#mixtrues
#    address, name, quantity, container, list_qt, list_sol
mixtures = [
dict(address =     outer_run_sol_1, name = stock_names["outer_run_solution_1"], volume = 12500, list_qt = [ v_stock_out, v_diluter_out ], list_sol = [stock_out_1, diluter_out_1], use = v_stock_out != 1), 
dict(address =     outer_run_sol_2, name = stock_names["outer_run_solution_2"], volume = 12500, list_qt = [ v_stock_out, v_diluter_out ], list_sol = [stock_out_2, diluter_out_2], use = v_stock_out != 1), 
dict(address =     inner_run_sol_1, name = stock_names["inner_run_solution_1"], volume = 12500, list_qt = [  v_stock_in,  v_diluter_in ], list_sol = [ stock_in_1,  diluter_in_1], use =  v_stock_in != 1), 
dict(address =     inner_run_sol_2, name = stock_names["inner_run_solution_2"], volume = 12500, list_qt = [  v_stock_in,  v_diluter_in ], list_sol = [ stock_in_2,  diluter_in_2], use =  v_stock_in != 1), 
dict(address = unused_sample_waste, name =     stock_names["washing_solution"], volume =   500, list_qt = [                           1], list_sol = [                stock_wash], use =             True),
dict(address =  sample_rinse_waste, name =     stock_names["washing_solution"], volume =   500, list_qt = [                           1], list_sol = [                stock_wash], use =             True),
dict(address =   first_rinse_waste, name =     stock_names["washing_solution"], volume =   500, list_qt = [                           1], list_sol = [                stock_wash], use =             True),
dict(address =  first_rinse_source, name =     stock_names["washing_solution"], volume =  1400, list_qt = [                           1], list_sol = [                stock_wash], use =             True) #1400
]

mxg = Mixing_Graph()

mover_tip   = Well_Address("6", "A1")
intent = Opentrons_HTTP_Communications.INTENT_SETUP 
# ---- start run
op.clear_run_if_needed()
op.create_run() # .... starting setup a new http script run
op.home()

op.pick_up_tip(pipette, mover_tip, intent)

for i in range(9):
    awell = op.well_by_address(Well_Address("10", "A1"))
    awell.used = True
    op.next_cuvette(pipette, "10")

op.drop_tip(pipette, mover_tip, intent)

op.clear_run_if_needed()


print(" ....................aaaaaaaaaaaaaaaaaaaand we are done!!!")


#================================================



