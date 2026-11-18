################################################################################
#   scripts wraps `solution_repository_manipulations.py` to edit 
#       a set of experiments 
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
#> sys.path.append("C:/Users/admin/Documents/Data/aikars/opentron/hld_ift_http/src") # robolab laptop
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

try:
    with open(f'{args.source}.bak', "w") as f:
        json.dump(params, f, indent = 2)
except Exception as e:
    print(str(e))
    exit()



from hld_ift_analysis_helpers.solution_repository_manipulations import SolutionRepositoryApp

SOLUTION_REPOSITORY_PATH = params["SOLUTION_REPOSITORY_PATH"]

app = SolutionRepositoryApp(SOLUTION_REPOSITORY_PATH)

k = input("Do you want to set main stocks??? [y/n] >>>")

if k == "" or k == "y":
    
    print("################################################################################")
    print(f'#{"assing stocks in scan_settings.json":^78}#')
    print("################################################################################")
    
    def make_me_pretty(a_str):
        return f'\n################################################################################\n{a_str:^78}\n################################################################################\n'
    
    conc = input("Stock concentration (default: 20.0)>>>")
    try:
        conc_val = float(conc)
    except:
        conc_val = 20.0
    
    
    outer_stock_1 = app.pick_a_solution(make_me_pretty("Select main outer stock 1 solution"))
    outer_stock_2 = app.pick_a_solution(make_me_pretty("Select main outer stock 2 solution"))
    
    outer_diluter_1 = app.pick_a_solution(make_me_pretty("Select outer diluter 1 solution"))
    outer_diluter_2 = app.pick_a_solution(make_me_pretty("Select outer diluter 2 solution"))
    
    inner_stock_1 = app.pick_a_solution(make_me_pretty("Select main inner stock 1 solution"))
    inner_stock_2 = app.pick_a_solution(make_me_pretty("Select main inner stock 2 solution"))
    
    inner_diluter_1 = app.pick_a_solution(make_me_pretty("Select inner diluter 1 solution"))
    inner_diluter_2 = app.pick_a_solution(make_me_pretty("Select inner diluter 2 solution"))
    
    
    print("selection:")
    print(conc_val)
    print(outer_stock_1)
    print(outer_stock_2)
    print(outer_diluter_1)
    print(outer_diluter_2)
    print(inner_stock_1)
    print(inner_stock_2)
    
    
    params["c_surfactant_outer_stock"] = conc_val
    params["c_surfactant_inner_stock"] = conc_val
    params["stocks"]["outer_stock_1"]   = params["stocks"]["outer_stock_1"] if outer_stock_1 is None else outer_stock_1  
    params["stocks"]["outer_stock_2"]   = params["stocks"]["outer_stock_2"] if outer_stock_2 is None else outer_stock_2  
    params["stocks"]["outer_diluter_1"] = params["stocks"]["outer_diluter_1"] if outer_diluter_1 is None else outer_diluter_1  
    params["stocks"]["outer_diluter_2"] = params["stocks"]["outer_diluter_2"] if outer_diluter_2 is None else outer_diluter_2  
    params["stocks"]["inner_stock_1"]   = params["stocks"]["inner_stock_1"] if inner_stock_1 is None else inner_stock_1  
    params["stocks"]["inner_stock_2"]   = params["stocks"]["inner_stock_2"] if inner_stock_2 is None else inner_stock_2  
    params["stocks"]["inner_diluter_1"] = params["stocks"]["inner_diluter_1"] if inner_diluter_1 is None else inner_diluter_1  
    params["stocks"]["inner_diluter_2"] = params["stocks"]["inner_diluter_2"] if inner_diluter_2 is None else inner_diluter_2  
    
    print(args.source)

k = input("Do you want to set up hld-ift run stocks??? [y/n] >>>")

if k == "" or k == "y":
     
    def get_run_conc_val(label, stock_conc_lbl, run_conc_lbl):
        conc_set = False
        while not conc_set:
            conc = input("HLD_IFT scan: surfactant concentration in {label}>>>")
            try:
                conc_val = float(conc)
                if conc_val <= params[stock_conc_lbl]:
                    params[run_conc_lbl] = conc_val
                    conc_set = True
                else:
                    print(f'A concentration of surfactant in a run solution cannot exceed a concentration in a stock ({params[stock_conc_lbl]})!!!')
            except:
                print(f'Use float value, should be less than concentration of surfactant in a stock({params[stock_conc_lbl]})!!!')
    
    #conc_set = False
    #while not conc_set:
    #    conc = input("HLD_IFT scan concentration >>>")
    #    try:
    #        conc_val = float(conc)
    #        if conc_val <= params["c_surfactant_stock"]:
    #            params["c_surfactant_experiment"] = conc_val
    #            conc_set = True
    #        else:
    #            print(f'A concentration of stock for an experiment cannot exceed a general stock ({params["c_surfactant_stock"]})!!!')
    #    except:
    #        print(f'Use float value, should be less than concentration of surfactant stock({params["c_surfactant_stock"]})!!!')

    get_run_conc_val("outer run solution", "c_surfactant_outer_stock", "c_surfactant_outer_run_solution")
    get_run_conc_val("inner run solution", "c_surfactant_inner_stock", "c_surfactant_inner_run_solution")

    
    def get_run_sol_name(stock_name, run_sol_name, label):
        k = input(f'Enter a name for {label}:\n   name of general stock: {params["stocks"][stock_name]}\n>>>')
        params["stocks"][run_sol_name] = k

    get_run_sol_name("outer_stock_1", "outer_run_solution_1", "outer run solution 1")
    get_run_sol_name("outer_stock_2", "outer_run_solution_2", "outer run solution 2")
    get_run_sol_name("inner_stock_1", "inner_run_solution_1", "inner run solution 1")
    get_run_sol_name("inner_stock_2", "inner_run_solution_2", "inner run solution 2")

    #k = input(f'Enter a name for run stock #1:\n   name of general stock: {params["stocks"]["outer_stock_1"]}\n>>>')
    #params["stocks"]["outer_run_solution_1"] = k

    #k = input(f'Enter a name for run stock #2:\n   name of general stock: {params["stocks"]["outer_stock_2"]}\n>>>')
    #params["stocks"]["outer_run_solution_2"] = k

    k = input(f'Enter a configuration name:\n   [date]_[conc]_[surfactant]_in_[oils]\n e.g. `2026-03-25_05g_surfA_in_C7C16`\n>>>')
    params["configurations"]["start"] = k
    params["configurations"]["end"] = f'{k}_run'


try:
    with open(args.source, "w") as f:
        json.dump(params, f, indent = 2)
except Exception as e:
    print(str(e))
    exit()

