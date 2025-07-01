################################################################################
# create representation of stock solutions for BrijL4 experiments
################################################################################
# /d/temp_data/scripts/generate_recepies_for_solutions.py
import sys
import functools
import json
import time
import numpy as np
import datetime
from scipy.optimize import minimize
import argparse

#sys.path.append("D:/projects/HLD_parameter_determination/hld_ift_http/src") # on office pc in miniconda
#sys.path.append("/mnt/d/projects/HLD_parameter_determination/hld_ift_http/src") # on office pc
sys.path.append("C:/Users/admin/Documents/Data/aikars/opentron/hld_ift_http/src") # robolab laptop
print("current contant of my python path\n: {c}".format(c = sys.path))


from hld_ift_http.opentrons_configs import ( 
                                            Opentrons_Configuration,
                                            Instrument_Configuration,
                                            Labware_Configuration,
                                            IFT_Cuvette_Stage_10mm_v1,
                                            Well_Address,
                                            Well_Configuration,
                                            Labware_Configuration_Loader
                                            )
from hld_ift_http.opentrons_http_comms import Opentrons_HTTP_Communications
from hld_ift_http.mixing_graph import Mixing_Graph
from hld_ift_http.opentrons_pp import Opentrons_PP
from hld_ift_http.compound_properties import Compound_Properties
from hld_ift_http.solution import Solution, Solution_Component
from hld_ift_http.solution_repository import Solution_Repository
from hld_ift_http.json_helpers import has_all_keys_required, missing_keys 
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

SOLUTION_REPOSITORY_PATH = params["solution_repository_path"]
   

### to init new repository use line below
#> rep = Solution_Repository()
### to open existing repository use line below
rep = Solution_Repository.fromJSON(file = SOLUTION_REPOSITORY_PATH)

# ================================================================================
#                                                           inputs 


def create_recepie_for(sol1, sol2, component, target, concentration_type, amount_needed = 1, method = "Nelder-Mead"):
    """
        function makes recipe for solution with a component at a given concentration
        :param Solution sol1: a solution to use
        :param Solution sol2: a solution to use
        :param str component: a name of component for which target concentration is specified
        :param str concentration_type: {"m/m": mass to mass of solution, "m/v": mass of component to volume of solution}, any other string defaults to "m/m"
        :param float amount_needed: mass of solution needed in grams
        :returns: Solution or None if recipe fails
    """
    params = {
            "m/v": {
                    "units": "g/ml",
                    "concentration": "m/v",
                    },
            "default": {
                    "units": "g/g",
                    "concentration": "m/m",
                        },
            }
    def temp_sol(x):
        return Solution.combine(
                                [x, 1-x],
                                ["mass", "mass"],
                                [sol1, sol2],
                                None,
                                ro_final = -1
                                )
    def optimize_by_weight_fraction(x):
        s = temp_sol(x[0])
        return abs(s.components[component].w - target)
    def optimize_by_m_v(x):
        s = temp_sol(x[0])
        return abs(s.components[component].w - target / s.ro)
    def msg_str(conc_type):
        params_use = params[conc_type]
        return f'optimizing for\n\tcomponent: `{component}`\n\tconcentration type: `{params_use["concentration"]}`\n\ttarget: {target} {params_use["units"]}'
    def binary_mix_recipe_dict_str(mixture, solute, solvent, m_solute, m_solvent):
        return '{\n' + \
        f'    \"mixture_type\": \"binary_mix\",\n' + \
        f'    \"name\": \"{mixture.name}_{datetime.datetime.now().strftime("%Y%m%d")}\",\n' + \
        f'    \"m_solute\":   _{m_solute:0.4f}_,\n' + \
        f'    \"m_solvent\":  _{m_solvent:0.4f}_,\n' + \
        f'    \"solvent\":   \"{solvent.name}\",\n' + \
        f'    \"solute\":    \"{solute.name}\",\n' + \
        f'    \"v_ro\":       _1.0_,\n' + \
        f'    \"m_ro_water\": _1.0_,\n' + \
        f'    \"m_ro\":       _{mixture.ro:0.5f}_,\n' + \
        f'    \"date\":       \"{datetime.datetime.now().strftime("%Y-%m-%d")}\",\n' + \
        '}'
 
    res = None
    if concentration_type == "m/v":
        print(msg_str(concentration_type))
        res = minimize(optimize_by_m_v, np.array([0.5]), method = method, bounds = [(0, 1)])
    else: # assume target is mass fraction
        print(msg_str("default"))
        res = minimize(optimize_by_weight_fraction, np.array([0.3]), method = method, bounds = [(0, 1)])

    if res is not None and res.success:
        x = res.x[0]
        resulting_mixture = temp_sol(x) 
        m_solute = x * amount_needed
        m_solvent = (1-x) * amount_needed
        indent_str = '    '  
        
        print(f'Quantity: {amount_needed:0.4f} g\n\nUse:\n{indent_str}{m_solute:0.4f} g of {sol1.name} and\n{indent_str}{m_solvent:0.4f} g of {sol2.name}')
        print(f'\n{indent_str}m({sol2.name})/m({sol1.name}): {(1 - x) / x: 0.6f}  g/g')

        print("\n------- dictionary for inclusion in solution repository -------\n")
        print(binary_mix_recipe_dict_str(resulting_mixture, sol1, sol2, m_solute, m_solvent)) 
        print(f"\n------- target solution representation -------\n\n{json.dumps(resulting_mixture.toDict(), indent = 2)}")
        return temp_sol(x)
    else:
        print('failed to create a recepie')
        print(res)
        return None

def mix(sol1, m1, sol2, m2, ro_final, name):
    sol = Solution.combine(
                            [m1, m2],
                            ["mass", "mass"],
                            [sol1, sol2],
                            None,
                            ro_final = ro_final
                            )
    sol.name = name
    return sol

def available_items():
    print("\n===============================================")
    print("\n--- solution names ---")
    print(rep.list_solution_names())
    print("\n--- component names ---")
    print(rep.list_components())
    print("===============================================\n")

# =========================================================
# add new compounds/solutions, if needed
# =========================================================
available_items()        

for it in params["compounds_to_add"]:
    rep.add_item(it)

available_items()

#print(json.dumps(rep.items["NovelTDA6"].toDict(), indent = 2))

# =========================================================
# recepies needed
# =========================================================
def recepie_wrapper(a_dict):
    print("\n========================================")
    create_recepie_for(
                    rep.items[a_dict["solution1"]],
                    rep.items[a_dict["solution2"]],
                    a_dict["component_to_target"],
                    a_dict["target_concentration"],
                    a_dict["concentration_type"],
                    a_dict["quantity"]
                    )

for r in params["recepies"]:
    recepie_wrapper(r)

print("\n===========================================================\n")

available_items()

#>with open(SOLUTION_REPOSITORY_PATH, "w") as f:
#>    rep.toJSON(file = f, sort_keys = True, indent = 2)




