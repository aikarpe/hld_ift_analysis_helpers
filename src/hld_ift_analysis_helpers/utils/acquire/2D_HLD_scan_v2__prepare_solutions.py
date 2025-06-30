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
from hld_ift_http.mixing_graph import Parent_Solution, Mixing_Vertice, Mixing_Graph
from hld_ift_http.solution_repository import Solution_Repository
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

# ............................................................ inputs
suffix_in = "hld_ift_experiment__blank"
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


c_surf_stock = params["c_surfactant_stock"]
c_surf_exp = params["c_surfactant_experiment"]
v_stock = c_surf_exp / c_surf_stock
v_solvent = (c_surf_stock - c_surf_exp) / c_surf_stock

stock_surf_oil_1  = Well_Address("9", "A1")
stock_surf_oil_2  = Well_Address("9", "A2")
stock_oil_1       = Well_Address("9", "A3")
stock_oil_2       = Well_Address("9", "A4")
stock_wt          = Well_Address("9", "A5")  


#================================================================================
#                              STOCK DETAILS 
#================================================================================
#inputs
#    address, soluton, quantity, container
stock_names = params["stocks"]
stocks = [
dict(address = stock_surf_oil_1, solution = stock_names["surfactant_in_oil_1"], volume = 12000), 
dict(address = stock_surf_oil_2, solution = stock_names["surfactant_in_oil_2"], volume = 12000), 
dict(address =      stock_oil_1, solution =               stock_names["oil_1"], volume = 12000), 
dict(address =      stock_oil_2, solution =               stock_names["oil_2"], volume = 12000),
dict(address =         stock_wt, solution =                            "water", volume = 12000) 
]

#find solution in repository
#from opentronpp get well with given address
#updates well with solution, volume

run_stock_surf_oil_1    = Well_Address("9", "B1")
run_stock_surf_oil_2    = Well_Address("9", "C1")
unused_sample_waste     = Well_Address("2", "A1")
sample_rinse_waste      = Well_Address("2", "A2")
first_rinse_waste       = Well_Address("2", "A3")
first_rinse_source      = Well_Address("2", "A4")

#================================================================================
#                              SOLUTIONS NEEDED
#================================================================================
#mixtrues
#    address, name, quantity, container, list_qt, list_sol
mixtures = [
dict(address = run_stock_surf_oil_1, name = stock_names["run_stock_surf_oil_1"], volume = 12500, list_qt = [ v_stock, v_solvent ], list_sol = [stock_surf_oil_1, stock_oil_1], use = v_stock != 1),
dict(address = run_stock_surf_oil_2, name = stock_names["run_stock_surf_oil_2"], volume = 12500, list_qt = [ v_stock, v_solvent ], list_sol = [stock_surf_oil_2, stock_oil_2], use = v_stock != 1), 
dict(address =  unused_sample_waste, name =                             "water", volume =   500, list_qt = [                   1], list_sol = [                     stock_wt], use =         True),
dict(address =   sample_rinse_waste, name =                             "water", volume =   500, list_qt = [                   1], list_sol = [                     stock_wt], use =         True),
dict(address =    first_rinse_waste, name =                             "water", volume =   500, list_qt = [                   1], list_sol = [                     stock_wt], use =         True),
dict(address =   first_rinse_source, name =                             "water", volume =  1400, list_qt = [                   1], list_sol = [                     stock_wt], use =         True) 
]

mxg = Mixing_Graph()

# ---- start run
op.clear_run_if_needed()
op.create_run() # .... starting setup a new http script run
op.home()

def update_stock_sol_info(**kwargs):
    well = op.well_by_address(kwargs["address"])
    well.used = True
    well.solution = rep.items[kwargs["solution"]]
    well.volume = kwargs["volume"]
    well.pipette = {}
    if (well.solution == None):
        well.volume = 0
        well.used = False


def add_to_mixing_graph(**kwargs):
    total = sum(kwargs["list_qt"])
    qt_ = list(map(lambda x: x / total, kwargs["list_qt"]))
    parents = list(map(lambda qt, sol: Parent_Solution(qt, sol), qt_, kwargs["list_sol"]))
    mxg.add_vertice(Mixing_Vertice(kwargs["address"], parents)) 

def execute_dilution(**kwargs):
    #@>kwargs2 = kwargs.copy()
    #@>kwargs2["repetitions"] = 1
    #@>kwargs2["mix_graph"] = mxg
    #@>kwargs2["pipette"] = pipette
    #@>kwargs2["a_well"
    #@>op.make_solution(**kwargs)
    #1>op.make_solution(mxg, pipette, kwargs["address"], kwargs["volume"], kwargs)
    op.make_solution(mxg, pipette, kwargs["address"], kwargs["volume"], repetitions = 1)

def add_to_repository(**kwargs):
    print(f'should use name: `{kwargs["name"]}`')
    rep.add_solution(op.well_by_address(kwargs["address"]).solution, kwargs["name"])

for st in stocks:
    update_stock_sol_info(**st)

first = True
for mix_input in mixtures:
    #if first:
    if mix_input[["use"]]:
        add_to_mixing_graph(**mix_input)
    #    first = False

first = True
for mix_input in mixtures:
    #if first:
    if mix_input[["use"]]:
        execute_dilution(**mix_input)
    #    first = False

first = True
for mix_input in mixtures:
    #if first:
    if mix_input[["use"]]:
        add_to_repository(**mix_input)
    #    first = False

# ---- end run here!!!
op.drop_tip_at_origin(pipette, Opentrons_HTTP_Communications.INTENT_SETUP)
op.clear_run_if_needed()

print(f'mixing graph now:\n {mxg.toJSON(indent = 2)}')

print(str(rep))

with open(SOLUTION_REPOSITORY_PATH, "w") as f:
    rep.toJSON(file = f, sort_keys = True, indent = 2)

print(" ....................aaaaaaaaaaaaaaaaaaaand we are done!!!")


