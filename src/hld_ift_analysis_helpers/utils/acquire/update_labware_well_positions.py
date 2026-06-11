################################################################################
# script provides 3 point check on labware definition and coordinat update
#   in definition file
#
# bits of useful stuff from other scripts
#
################################################################################
# having matrix of known index inputs M_inp and known output coordinates M_out
#
# perform Singular value decomposition (SVD)
# M_inp = U * D * t(V), 
#   where following is true:
#       U * t(U) = I
#       V * t(V) = I
# X <- transform matrix
# X * M_inp = M_out
# X * U * D * t(V) = M_out
# X * U * D * t(V) * V * inv(D) * t(U) = M_out * V * inv(D) * t(U) 
# X = M_out * V * inv(D) * t(U) 


#
#```{R}
#require(svd)
#out_M <- t(matrix(c(4.3, 4.3, 106.2, 62.2, 5.7, 62.2,0,0,0,1,1,1), ncol = 4))
#inp_M <- t(matrix(c(0, 0, 4, 0, 2, 0,0,0,0,1,1,1), ncol = 4))
#all_inputs_M <- t(as.matrix(expand.grid(x = seq(0,4), y = seq(0,2), z = 0, c = 1)))
#
#dec_s <- svd(inp_M)
#trans_M <- out_M %*% dec_s$v %*% solve(diag(dec_s$d)) %*% t(dec_s$u) 
#all_outputs_M <- trans_M %*% all_inputs_M
#```

#=== def subset_me(x):
#===     return x[:,:]
#=== inp_M = subset_me(np.array(
#===         [[0, 0, 4],                   #[[0, 0, 4, 4], 
#===          [0, 2, 0],                   # [0, 2, 0, 2], 
#===          [0, 0, 0],                   # [0, 0, 0, 0], 
#===          [1, 1, 1]]                   # [1, 1, 1, 1]] 
#===         ))
#=== out_M = subset_me(np.array(
#===         [[ 4.3, 4.3,106.2],           #[[ 4.3, 4.3,106.2,105.7],
#===          [62.2, 5.7, 62.2],           # [62.2, 5.7, 62.2,  5.3],
#===          [ 0.0, 0.0,  0.0],           # [ 0.0, 0.0,  0.0,  0.0],
#===          [ 1.0, 1.0,  1.0]]           # [ 1.0, 1.0,  1.0,  1.0]]
#===         ))
#=== 
#=== all_inputs_M = np.array(
#===         [[0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4],
#===          [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
#===          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#===          [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
#===         )[:,:]
#=== 
#=== output = scipy.linalg.svd(inp_M, full_matrices = False)
#=== #output = scipy.linalg.svd(inp_M.T, full_matrices = False)
#=== U = output[0]
#=== D = scipy.sparse.diags_array(output[1]).toarray()
#=== V = output[2].T
#=== 
#=== # X = M_out * V * inv(D) * t(U) 
#=== TRANSFORM = out_M @ V @ scipy.linalg.inv(D) @ U.T
#=== TRANSFORM @ all_inputs_M
#=== 



################################################################################
import sys
import os
import functools
import json
import time
import argparse
import datetime
import scipy.linalg
import numpy as np
import codecs

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
import hld_ift_http.errors

debug = True

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
suffix_in = params["configurations"]["blank"]
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


#__ c_surf_stock = params["c_surfactant_stock"]
#__ c_surf_exp = params["c_surfactant_experiment"]
#__ v_stock = c_surf_exp / c_surf_stock
#__ v_solvent = (c_surf_stock - c_surf_exp) / c_surf_stock
#__ 
#__ stock_surf_oil_1  = Well_Address("9", "A1")
#__ stock_surf_oil_2  = Well_Address("9", "A2")
#__ stock_oil_1       = Well_Address("9", "A3")
#__ stock_oil_2       = Well_Address("9", "A4")
#__ stock_wt          = Well_Address("9", "A5")  


#================================================================================
#                              STOCK DETAILS 
#================================================================================
#inputs
#    address, soluton, quantity, container
#__ stock_names = params["stocks"]
#__ stocks = [
#__ dict(address = stock_surf_oil_1, solution = stock_names["surfactant_in_oil_1"], volume = 12000), 
#__ dict(address = stock_surf_oil_2, solution = stock_names["surfactant_in_oil_2"], volume = 12000), 
#__ dict(address =      stock_oil_1, solution =               stock_names["oil_1"], volume = 12000), 
#__ dict(address =      stock_oil_2, solution =               stock_names["oil_2"], volume = 12000),
#__ dict(address =         stock_wt, solution =                            "water", volume = 12000) 
#__ ]

#find solution in repository
#from opentronpp get well with given address
#updates well with solution, volume

#__ run_stock_surf_oil_1    = Well_Address("9", "B1")
#__ run_stock_surf_oil_2    = Well_Address("9", "C1")
#__ unused_sample_waste     = Well_Address("2", "A1")
#__ sample_rinse_waste      = Well_Address("2", "A2")
#__ first_rinse_waste       = Well_Address("2", "A3")
#__ first_rinse_source      = Well_Address("2", "A4")

#================================================================================
#                              SOLUTIONS NEEDED
#================================================================================
#mixtrues
#    address, name, quantity, container, list_qt, list_sol
#__ mixtures = [
#__ dict(address = run_stock_surf_oil_1, name = stock_names["run_stock_surf_oil_1"], volume = 12500, list_qt = [ v_stock, v_solvent ], list_sol = [stock_surf_oil_1, stock_oil_1], use = v_stock != 1),
#__ dict(address = run_stock_surf_oil_2, name = stock_names["run_stock_surf_oil_2"], volume = 12500, list_qt = [ v_stock, v_solvent ], list_sol = [stock_surf_oil_2, stock_oil_2], use = v_stock != 1), 
#__ dict(address =  unused_sample_waste, name =                             "water", volume =   500, list_qt = [                   1], list_sol = [                     stock_wt], use =         True),
#__ dict(address =   sample_rinse_waste, name =                             "water", volume =   500, list_qt = [                   1], list_sol = [                     stock_wt], use =         True),
#__ dict(address =    first_rinse_waste, name =                             "water", volume =   500, list_qt = [                   1], list_sol = [                     stock_wt], use =         True),
#__ dict(address =   first_rinse_source, name =                             "water", volume =  1400, list_qt = [                   1], list_sol = [                     stock_wt], use =         True) 
#__ ]
#__ 
#__ mxg = Mixing_Graph()

# ---- start run
op.clear_run_if_needed()
op.create_run() # .... starting setup a new http script run
op.home()

#__ def update_stock_sol_info(**kwargs):
#__     well = op.well_by_address(kwargs["address"])
#__     well.used = True
#__     well.solution = rep.items[kwargs["solution"]]
#__     well.volume = kwargs["volume"]
#__     well.pipette = {}
#__     if (well.solution == None):
#__         well.volume = 0
#__         well.used = False
#__ 
#__ 
#__ def add_to_mixing_graph(**kwargs):
#__     total = sum(kwargs["list_qt"])
#__     qt_ = list(map(lambda x: x / total, kwargs["list_qt"]))
#__     parents = list(map(lambda qt, sol: Parent_Solution(qt, sol), qt_, kwargs["list_sol"]))
#__     mxg.add_vertice(Mixing_Vertice(kwargs["address"], parents)) 
#__ 
#__ def execute_dilution(**kwargs):
#__     #@>kwargs2 = kwargs.copy()
#__     #@>kwargs2["repetitions"] = 1
#__     #@>kwargs2["mix_graph"] = mxg
#__     #@>kwargs2["pipette"] = pipette
#__     #@>kwargs2["a_well"
#__     #@>op.make_solution(**kwargs)
#__     #1>op.make_solution(mxg, pipette, kwargs["address"], kwargs["volume"], kwargs)
#__     op.make_solution(mxg, pipette, kwargs["address"], kwargs["volume"], repetitions = 1)
#__ 
#__ def add_to_repository(**kwargs):
#__     print(f'should use name: `{kwargs["name"]}`')
#__     rep.add_solution(op.well_by_address(kwargs["address"]).solution, kwargs["name"])
#__ 
#__ for st in stocks:
#__     update_stock_sol_info(**st)
#__ 
#__ first = True
#__ for mix_input in mixtures:
#__     #if first:
#__     if mix_input[["use"]]:
#__         add_to_mixing_graph(**mix_input)
#__     #    first = False
#__ 
#__ first = True
#__ for mix_input in mixtures:
#__     #if first:
#__     if mix_input[["use"]]:
#__         execute_dilution(**mix_input)
#__     #    first = False
#__ 
#__ first = True
#__ for mix_input in mixtures:
#__     #if first:
#__     if mix_input[["use"]]:
#__         add_to_repository(**mix_input)
#__     #    first = False
#__ 
#__ # ---- end run here!!!

op.pick_up_tip_safely_for(pipette, Well_Address("2", "A1"), Opentrons_HTTP_Communications.INTENT_SETUP)


# my adjustment changes begin!!!

def choose_slot_to_modify():
    allowed_slot_values = '0123456789'
    print("Choose slot to modify (0: exit; `1` to `9` to select a slot)") 
    k = input(">>>")
    out = k[0] if len(k)>0 and k[0] in allowed_slot_values else ""
    return out

def bak_file_path(path, suffix):
    return f'{path}.{suffix}.bak'

def user_suffix():
    print("Enter suffix to store current version of labware definition file")
    print("``(empty input) defaults to YYYYMMDD_HHMM type of string of current date and time")
    k = input("your choice ==>")
    suffix = datetime.datetime.now().strftime("%Y%m%d_%H%M") if k == "" else k
    return suffix

def message_user_on_current_choice(path):
    print("old labware definition is backed up here:")
    print("   folder:")
    x = os.path.split(path)
    print(f'        {x[0]}')
    print("   name:")
    print(f'        {x[1]}')
    
def message_user_canceled_adjustment():
    print("location adjustment was canceled!!!")
    print("    ... moving on ...")
    
def modify_location():
    step_small = 0.1
    step_large = 1.0
    choices = dict(
                l = dict(x = 1, y = 0, z = 0),
                j = dict(x = -1, y = 0, z = 0),
                k = dict(x = 0, y = -1, z = 0),
                i = dict(x = 0, y = 1, z = 0),
                e = dict(x = 0, y = 0, z = -1),
                d = dict(x = 0, y = 0, z = 1),
                r = dict(x = 0, y = 0, z = 0)
                )
    status_msg = dict(a = "accept", z = "cancel", r = "wait", m = "move")

    print("========================================================")
    print("| Movements:                                           |")
    print("|     i: away (depth)                                  |")
    print("|     k: closer (depth)                                |")
    print("|     j: left                                          |")
    print("|     l: right                                         |")
    print("|     e: up                                            |")
    print("|     d: down                                          |")
    print("|                                                      |")
    print("| step size:                                           |")
    print("|     lowercase key: 0.1 mm                            |")
    print("|     uppercase key:   1 mm                            |")
    print("|                                                      |")
    print("| Status:                                              |")
    print("|     a: accept changes                                |")
    print("|     z: cancel, discard changes                       |")
    print("|                                                      |")
    print("|------------------------------------------------------|")
    print("| !!! NO SAFETY FEATURES PRESENT, BE CAREFUL !!!       |")
    print("|------------------------------------------------------|")
    print("|       [e]            [i]                             |")
    print("|  [a]   [d]         [j][k][l]                         |")
    print("|   [z]                                                |")
    print("|------------------------------------------------------|")
    print("========================================================")
    k = input(">>>")
    status = "r" if len(k) < 1 else k[0]
    step_size = step_small if ord(status) > 96 else step_large 

    status = status.lower()

    status = status if status in "azedijklr" else "r"

    valid_multi_horizontal = len(k) >=2 and \
                                            k[1] in "123" and \
                                            status in "jikl"
    valid_multi_vertical = len(k) >=2 and \
                                            k[1] in "123456789" and \
                                            status in "ed"

    multiplier = float(k[1]) if valid_multi_vertical or valid_multi_horizontal else 1.0


    if status in ["a", "z", "r"]:
        return dict(status = status_msg[status], delta = None)
    else:
        ch_use = choices[status]
        return dict(
                    status = status_msg["m"],
                    delta = dict(
                        x = ch_use["x"] * step_size * multiplier,
                        y = ch_use["y"] * step_size * multiplier,
                        z = ch_use["z"] * step_size * multiplier
                    ))

def find_location(well_loc):
    delta = dict(x = 0.0, y = 0.0, z = 0.0)
    intent = Opentrons_HTTP_Communications.INTENT_SETUP 
    speed_init = 100
    speed = 10
    op.move_to_xyz(pipette, well_loc, delta, intent, speed_init, False) 

    active = True

    while active:

        out =  modify_location()

        status = out["status"]

        #status_msg = dict(a = "accept", z = "cancel", r = "wait", m = "move")
        if status == "accept":
            #calc_abs_pos(delta)
            
            return dict(status = "accept", delta = delta) #calc_abs_pos(delta))
        
        elif status == "cancel":
            return out
        elif status == "move":
            new_increment = out["delta"]
            delta["x"] = delta["x"] + new_increment["x"]
            delta["y"] = delta["y"] + new_increment["y"]
            delta["z"] = delta["z"] + new_increment["z"]
            op.move_to_xyz(pipette, well_loc, delta, intent, speed, True) 
        else:
            print(".")

def location_w_offset(loc_dict, delta_dict):
        return dict(
                    x = loc_dict["x"] + delta_dict["x"],
                    y = loc_dict["y"] + delta_dict["y"]
                    )

        
running = True

while running:
    choice = choose_slot_to_modify()

    if choice == "0":
        running = False
        continue

    slot_use = choice
    lw = op.labware_by_slot(slot_use)
    folder = lw.source_folder
    name = f'{lw.load_name}.json'
    load_path = os.path.join(folder, name)


    with codecs.open(load_path, "r", "utf-8") as f:
        labware_def = json.load(f)

    #with open(load_path, "r") as f:
    #    labware_def = json.load(f)
    

    #save copy of original definition
    suffix = user_suffix()
    original_definition_path = bak_file_path(load_path, suffix)
    with open(original_definition_path, "w", encoding = "utf-8") as f:
        json.dump(labware_def, f, ensure_ascii = False)
    message_user_on_current_choice(original_definition_path)


    ordering = labware_def["ordering"]
    

    N_outer = len(ordering)

    locations = dict() 

    if N_outer > 0:
        #>take 1st element of 1st group ==> x1,y1
        well_addr1 = ordering[0][0]
        output = find_location(Well_Address(slot_use, well_addr1))
        if output["status"] == "accept":
            locations["1"] = location_w_offset(labware_def["wells"][well_addr1], output["delta"])
            locations["1"]["index_outer"] = 0
            locations["1"]["index_inner"] = 0
        else:
            message_user_canceled_adjustment()
            continue
    
    if N_outer > 1:
        #>take 1st element of last group == x2,y2
        #>determine outer coodinate variable (x or y)    
        well_addr1 = ordering[N_outer - 1][0]
        output = find_location(Well_Address(slot_use, well_addr1))

        if output["status"] == "accept":
            locations["2"] = location_w_offset(labware_def["wells"][well_addr1], output["delta"])
            locations["2"]["index_outer"] = N_outer - 1
            locations["2"]["index_inner"] = 0
        else:
            message_user_canceled_adjustment()
            continue
    
    
    N_inner = len(ordering[0])

    if N_inner > 1:
        #>take last element of first group ==> x3, y3
        #>determine inner coordinate variable
        well_addr1 = ordering[0][N_inner - 1]
        output = find_location(Well_Address(slot_use, well_addr1))

        if output["status"] == "accept":
            locations["3"] = location_w_offset(labware_def["wells"][well_addr1], output["delta"])
            locations["3"]["index_outer"] = 0
            locations["3"]["index_inner"] = N_inner - 1
        else:
            message_user_canceled_adjustment()
            continue

    
    well_locations = np.array([[ locations[loc]["x"], locations[loc]["y"],0,1] for loc in locations ]).T
    index_locations = np.array([ [float(locations[loc]["index_outer"]), float(locations[loc]["index_inner"]),0,1] for loc in locations ]).T

    
    all_well_names = []
    all_index_loc = []
    for io, grp in enumerate(ordering):
        for ii, elem in enumerate(grp):
            all_index_loc.append([float(io), float(ii), 0.0, 1.0])
            all_well_names.append(elem)

    all_index_locations = np.array(all_index_loc).T

    output = scipy.linalg.svd(index_locations, full_matrices = False)
    U = output[0]
    D = scipy.sparse.diags_array(output[1]).toarray()
    V = output[2].T
    
    if debug:
        print("\nindex_locations")
        print(index_locations)

        print("\nwell_locations")
        print(well_locations)

        print("\nall_index_locations")
        print(all_index_locations)

        print("\nU")
        print(U)

        print("\nD")
        print(D)

        print("\nV")
        print(V)


    # X = M_out * V * inv(D) * t(U) 
    TRANSFORM = well_locations @ V @ scipy.linalg.inv(D) @ U.T

    if debug:
        print("\nTRANSFORM")
        print(TRANSFORM)

    corrected_well_locations = TRANSFORM @ all_index_locations

    if debug:
        print("\ncorrected_well_locations")
        print(corrected_well_locations)


    #_edit_ find  

   

    #> coord_lbls = set("x", "y")
    #> if (outer_var in coord_lbls and inner_var in coord_lbls and  outer_var != inner_var)
    #>     outer_values(p1, p2)
    #>     inner_values(p1, p3)
    #> 
    #> elif outer_val in  coord_lbls and inner_var not in coord_lbls:
    #>     outer_values(p1, p2)
    #>     inner_var = coord_lbls.difference(set(outer_val))[0]
    #>     inner_values = [(p1[inner_var] + p2[inner_var]) / 2]
    #> 
    #> elif inner_val in coord_lbls and outer_var not in coord_lbls:
    #>     inner_values(p1,p3)
    #>     outer_var = coord_lbls.difference(set(inner_val))[0]
    #>     outer_values = [(p1[outer_var] + p3[outer_var]) / 2]
    #> 
    #> elif outer_var not in coord_lbls and inner_var not in coord_lbls:
    #>     inner_var = "x"
    #>     outer_var = "y"
    #>     inner_values = [p1[inner_var]]
    #>     outer_values = [p1[outer_var]]
    #> else:
    #>     !!!mistake!!!
    
    for an_index, well_nm in enumerate(all_well_names):
        dict_to_edit = labware_def["wells"][well_nm]
        dict_to_edit["x"] = corrected_well_locations[0][an_index]
        dict_to_edit["y"] = corrected_well_locations[1][an_index]

    with open(load_path, "w", encoding = "utf-8") as f:
        json.dump(labware_def, f, ensure_ascii = False)



op.drop_tip_at_origin(pipette, Opentrons_HTTP_Communications.INTENT_SETUP)


op.clear_run_if_needed()

#> print(f'mixing graph now:\n {mxg.toJSON(indent = 2)}')

#> print(str(rep))

#>with open(SOLUTION_REPOSITORY_PATH, "w") as f:
#>    rep.toJSON(file = f, sort_keys = True, indent = 2)

print(" ....................aaaaaaaaaaaaaaaaaaaand we are done!!!")



