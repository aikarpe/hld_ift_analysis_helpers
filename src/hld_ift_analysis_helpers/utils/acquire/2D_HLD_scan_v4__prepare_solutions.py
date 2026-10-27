################################################################################
# a script to create run stocks for scan with water soluble and oil soluble surfactants
#   Script assumes that:
#       one surfactant is in oil; another surfactant is in aqueous phase
#       each surfactant has two main stocks and each main stock has associated dilution liquid
#
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
dict(address =  first_rinse_source, name =     stock_names["washing_solution"], volume =  1400, list_qt = [                           1], list_sol = [                stock_wash], use =             True) 
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
    if mix_input["use"]:
        add_to_mixing_graph(**mix_input)
    #    first = False

# ............................................................ pretty print of config

pps = {
    'to_extract': dict(
            	volume = "volume",
            	used = "used",
            	available = "available",
            	solution_name = solution_name_str,
                address = well_address_str
	            ),
    'filter_query': 'role != ""',
    'conversion_fn': lambda address,volume,solution_name,role,available,used: f'{address: >4}: {volume: 8d} uL, [{role: >40}]: `{solution_name}`',
    'lst_roles': {
        '2/A1': "out         SMPL WST",
        '2/A2': "out         WASH1 WST",
        '2/A3': "out         WASH2 WST",
        '2/A4': "out         WASH1",
        '9/A1': "out         OUTER RUN 1",
        '9/A2': "out         OUTER RUN 2",
        '9/A3': "out         INNER RUN 1",
        '9/A4': "out         INNER RUN 2",
        '9/B1': "in 12500uL  OUTER STCK 1",
        '9/B2': "in 12500uL  OUTER STCK 2",
        '9/B3': "in 12500uL  INNER STCK 1",
        '9/B4': "in 12500uL  INNER STCK 2",
        '9/C1': "in 12500uL  OUTER DIL 1",
        '9/C2': "in 12500uL  OUTER DIL 2",
        '9/C3': "in 12500uL  INNER DIL 1",
        '9/C4': "in 12500uL  INNER DIL 2",
        '9/C5': "in 12500uL  WASH STCK"
        }
}

pretty_config_make(
        op,
        pps['to_extract'],
        pps['filter_query'],
        pps['conversion_fn'],
        pps['lst_roles'],
        wait_for_user = True)

# ........................................ pretty print of config END


first = True
for mix_input in mixtures:
    #if first:
    if mix_input["use"]:
        execute_dilution(**mix_input)
    #    first = False

first = True
for mix_input in mixtures:
    #if first:
    #if mix_input["use"]:
    add_to_repository(**mix_input)
    #    first = False

# ---- end run here!!!
op.drop_tip_at_origin(pipette, Opentrons_HTTP_Communications.INTENT_SETUP)

#op.discard_used_tips(pipette)

op.clear_run_if_needed()

print(f'mixing graph now:\n {mxg.toJSON(indent = 2)}')

print(str(rep))

with open(SOLUTION_REPOSITORY_PATH, "w") as f:
    rep.toJSON(file = f, sort_keys = True, indent = 2)

print(" ....................aaaaaaaaaaaaaaaaaaaand we are done!!!")


#================================================


