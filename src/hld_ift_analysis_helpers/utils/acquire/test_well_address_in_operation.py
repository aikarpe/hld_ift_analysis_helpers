################################################################################
#   test `in` operator for well_address
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

from hld_ift_analysis_helpers.utilities import free_disk_space_check

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
from hld_ift_http.opentron_well_status import pretty_config_make, solution_name_str, well_address_str 
import hld_ift_http.errors


outer_run_sol_1 = Well_Address("9", "A1") 
outer_run_sol_2 = Well_Address("9", "A2")
inner_run_sol_1 = Well_Address("9", "A3")
inner_run_sol_2 = Well_Address("9", "A4")

all = [ outer_run_sol_1, outer_run_sol_2, inner_run_sol_1, inner_run_sol_2 ]
some1 = [ outer_run_sol_2, inner_run_sol_1, inner_run_sol_2 ]
some2 = [ outer_run_sol_1, inner_run_sol_1, inner_run_sol_2 ]
some3 = [ outer_run_sol_1, outer_run_sol_2, inner_run_sol_2 ]

def test_variable(var, lbl):
    print(f"testing `{lbl}` in all, some1, some2, some3")
    for aset,albl in zip([all, some1, some2, some3],["all", "some1", "some2", "some3"]):
        print(f'testing {lbl} in {albl}: {var in aset}')	

test_variable(outer_run_sol_1, "outer_run_sol_1")
test_variable(outer_run_sol_2, "outer_run_sol_2")
test_variable(inner_run_sol_1, "inner_run_sol_1")
test_variable(inner_run_sol_2, "inner_run_sol_2")

