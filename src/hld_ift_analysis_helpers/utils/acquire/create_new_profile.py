# script creates scan and solution prep profiles for new surfactant
#
#   1. create surfactant folder
#   1.1 create config folder
#   2. from templates:
#       - copy blank config
#       - copy solution repository
#       - create `command_prompt.md` with right cd command <folder_name>
#       - create solution prep json:
#           modify path to repository
#           create dictionary <ro, name_string>
#           component name to name_string
#       - create scan json:
#           - create records for folders, logs, etc
#           - create stock solution names
#           - create generic names for run stocks


import os
import json
import shutil

params = dict(
            "surfactant_label": "Novel810-3.5",
            "folder_name": "Novel_810-3.5_v2_test",
            "surfactant_pretty_name": "Novel 810-3.5",
            )


template_source = "__add__"
root = "__add__"

files_to_copy = [
                "command_prompt_bits.md", 
                "scan_settings.json",
                "recipes_solutions.json",
                "config/solution_repository.json",
                "config/config_hld_ift_experiment__blank__opentron_pp.json",
                "config/config_hld_ift_experiment__blank__execute_measurement.json",
                "config/config_hld_ift_experiment__blank__execute_measurement__no_wash.json",
                "config/config_hld_ift_experiment__blank__execute_measurement__wash.json",
                "config/config_hld_ift_experiment__blank__mixing_graph.json"
                ]

project_root = os.path.join(root, folder_name) 
project_config = os.path.join(project_root, "config")
project_scripts = os.path.join(project_root, "scripts")
blank_config_path = os.path.join(project_config, "config_hld_ift_experiment__blank__opentron_pp.json") 
cmd_prompt_file = os.path.join(project_root, files_to_copy[0])
scan_settings_json = os.path.join(project_root, files_to_copy[1])
recipes_json = os.path.join(project_root, files_to_copy[2])
log_path = os.path.join(project_root, "log.log")
solution_repository_path = os.path.join(project_config, "solution_repository.json"

def modify_blank_profile(log_path):
    with open(blank_config_path, "r") as f:
        data = json.load(f)

    data["opentrons_api"]["log_path"] = log_path

    with open(blank_config_path, "w") as f:
        json.dump(data, f)

def copy_file(file_relative_path):
    shutil.copy(os.path.join(template_source, file_relative_path), os.path.join(project_root, file_relative_path))

def modify_command_prompt_bits():
    with open(cmd_prompt_file, "r") as fi:
        content = fi.read()

    lines = content.split("\n")

    lines[0] = f"cd {project_root}".replace("/", "\\")

    with open(cmd_prompt_file, "w") as fo:
        fo.writelines(lines)

def create_recipes():
    with open(recipes_json, "r") as fi:
        data = json.load(fi)

    data["solution_repository_path"] = solution_repository_path
    data["compounds_to_add"] = [ dict(
                                        mixture_type = "pure_compound",
                                        ro = params["ro"],
                                        name = params["surfactant_label"],
                                        label = params["surfactant_label"]
                                        )]
    def make_recipe(solvent):
        return dict(
                    solution1 = params["surfactant_label"],
                    solution2 = solvent,
                    component_to_target = params["surfactant_label"],
                    target_concentration = 0.2,
                    concentration_type = "m/v",
                    quantity = 50
                    )

    data["recipes"] = list(map(make_recipe, ["heptane", "hexadecane"]))
    
    with open(recipes_json, "w") as fo:
        json.dump(data, fo)
       

def create_scan_settings():
    with open(scan_settings_json, "r") as fi:
        data = json.load(fi)

    def surf_sol_name(conc_str, solvent):
        return f'{conc_str}g/100ml {params["surfactant_pretty_name"]} in {solvent} __date__'
    def config_name(conc_str):
        return f'2025-MM-DD_{conc_str}.00g_{params["surfactant_label"]}_C7C16_NaCl'

    def_conc_exp = 5
    data["DATA_PATH"] = project_root
    data["LOG_PATH"] = log_path
    data["CONFIG_PATH"] = project_config
    data["SOLUTION_REPOSITORY_PATH"] = solution_repository_path
        
    data["c_surfactant_experiment"] = def_cond_exp
    data["c_surfactant_stock"] = 20
    data["stocks"] = dict(
        surfactant_in_oil_1 = surf_sol_name("20", "heptane"),
        surfactant_in_oil_2 = surf_sol_name("20", "hexadecane"), 
        oil_1 = "heptane",
        oil_2 = "hexadecane",
        stock_aqueous_1 = "water",
        stock_aqueous_2 = "30g_NaCl_in_100mL_water_20250225",
        run_stock_surf_oil_1 = surf_sol_name(f'{def_conc_exp:02d}', "heptane"),
        run_stock_surf_oil_2 = surf_sol_name(f'{def_conc_exp:02d}', "hexadecane")
        )
    },
    data["configurations"] = dict(
                                blank = "hld_ift_experiment__blank",
                                start = config_name(f'{def_conc_exp:02d}'),
                                end = config_name(f'{def_conc_exp:02d}') + "_001",
                                to_reuse = dict(
                                                name = "hld_ift_experiment__blank",
                                                content = []
                                                ))
    data["scan"] = dict(
        experiment_metadata = dict(
            description = f'{params["surfactant_pretty_name"]} {def_conc_exp:02d}g/100ml oil (C7/C16)',
            needle_dia = 0.312,
            oil = "C7 to C16",
            measurement = "pulsed, 20 steps, 6 s pause",
            scan_type = "2D scan: salinity, oil"
            ),
        n_expansions = 3,
        n_approximation = 0,
        number_of_oil_points = 6,
        oil_volume = 3000,
        scan_type = "linear"
        )

    with open(scan_settings_json, "w") as fo:
        json.dump(data, fo)



os.makedirs(project_root)
os.makedirs(project_config)
os.makedirs(project_scripts)

for f in files_to_copy:
    copy_file(f)

modify_blank_profile(log_path)
modify_command_prompt_bits()
create_recipes()
create_scan_settings()
