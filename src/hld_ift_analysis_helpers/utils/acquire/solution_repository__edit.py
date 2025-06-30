################################################################################
# create representation of stock solutions for BrijL4 experiments
################################################################################
import sys
import functools
import json
import time
import argparse

sys.path.append("D:/projects/HLD_parameter_determination/hld_ift_http/src") # on office pc in miniconda
#sys.path.append("/mnt/d/projects/HLD_parameter_determination/hld_ift_http/src") # on office pc
#sys.path.append("C:/Users/admin/Documents/Data/aikars/opentron/hld_ift_http/src") # robolab laptop
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
#rep = Solution_Repository()
### to open existing repository use line below
rep = Solution_Repository.fromJSON(file = SOLUTION_REPOSITORY_PATH)

def available_items():
    print("\n===============================================")
    print("\n--- solution names ---")
    print(rep.list_solution_names())
    print("\n--- component names ---")
    print(rep.list_components())
    print("===============================================\n")

for it in params["compounds_to_add"]:
    rep.add_item(it)

print(str(rep))

available_items()

with open(SOLUTION_REPOSITORY_PATH, "w") as f:
    rep.toJSON(file = f, sort_keys = True, indent = 2)
