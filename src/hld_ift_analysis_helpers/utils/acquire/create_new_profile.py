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

# changes needed
#   inputs:
#       source folder
#       destination folder
#       solution_repository (if not in source)
#       
#   warning:
#       if destination folder exists and/or ...other conditions specify...
#
#   files to copy:
#       "command_prompt_bits.md", :: edit cd path line
#       "scan_settings.json", :: edit settings
#       "recipes_solutions.json",
#       files 
#               in ./config and
#               contain hld_ift in their name
#               contain _blank_ in their name
#   
#   edit cd path line:
#       read file line by line
#       if line starts with `cd ...a path...` change it to `cd source_folder`
#           path delimiters set to `\`
#       else 
#           leave line like is
#       write everything to source_folder\command_prompt_bits.md
#
#   edit settings:
#       "DATA_PATH"  <- source_folder
#       "LOG_PATH"   <- source_folder/log.log
#       "CONFIG_PATH" <- source_folder/config
#       "SOLUTION_REPOSITORY_PATH" <- solution repository if solution repository is not None else source_folder/solution_repository.json
#
#   edit configurations:
#       configurations/blank <- common blank name!!!
#       configurations/start <- __template__
#       configurations/end <- __template__
#       configurations/to_reuse/name <- configurations/blank
#       configurations/to_reuse/content <- []
#
#   edit stocks:
#       run stocks <- __edit_template__
#   edit scan:
#       scan/experiment_metadata/destination <- __edit_template__
#       rest as is
#
#   common blank name(files):
#       for each files name:
#           remove config_ from start
#           remove everything after `...blank`
#       find unique
#       if more than one unique string ask which one to use


# ================================================================================
# \                                idea for wizard                               \  
# ================================================================================
# takes through the whole process 
#   uses previous folder as an input
#   can choose random solution_rep
#   creates and adds solution to rep
#   creates whole folder structure
# ================================================================================

# +--- frame 1: source ---------------------------------------------------------+
# \ path: [__browser_1___]                                                      \
# \                                                                             \
# \ [ ] use different solution repository                                       \
# \ path_repository: [__browser_2___]                                           \
# +-----------------------------------------------------------------------------+

# frame 1
# __browser_1___ 
#           defaults to general scan folder
#           needs to check if path points to an experiment (has config, settings, etc)
# __browser_2___
#           defaults to path_to_rep in selected folder
#           check if file is solution_repository

# +--- frame 2: target ---------------------------------------------------------+
# \ path: [__browser_3___]                                                      \
# \                                                                             \
# \ name: [__folder_name_1__]                                                   \
# \                                                                             \
# \                                                                             \
# \ [ ] overwrite stuff if folder already exists                                \
# \                                                                             \
# +-----------------------------------------------------------------------------+

# __browser_3___ 
#           should be an existing folder
# __folder_name_1__
#           should be valid folder name
# overwrite
#           enable if __browser_3___/__folder_name_1__ already exists and is not empty
#           give warnign to user

# +--- frame 3: stocks ---------------------------------------------------------+
# \ surfactant: [__choose_one_1__]       [[ add new surfactant + ]]                                               \
# \                                                                             \
# \ oil 1: [__choose_oil_1__]                                                   \
# \ oil 2: [__choose_oil_2__]                                                   \
# \ aqueous 1: [__choose_aq_1__]                                                \
# \ aqueous 2: [__choose_aq_2__]                                                \
# \                                                                             \
# +-----------------------------------------------------------------------------+

# __choose_one_1__
#           all simple substances with 1 component withing sol_rep
# !!!need some idea how to select solutions here without overwelming user!!!          
# __choose_oil_1__, __choose_oil_2__, __choose_aq_1__, __choose_aq_2__
#           default oils and aq stuff from previous experiment!!!


# +--- frame X: add new surfactant ---------------------------------------------+
# \                                                                             \
# \ surfactant name: [__surfactant_name__]                                      \
# \ surfactant desnity: [__denisty_1__]                                         \
# \                                                                             \
# +-----------------------------------------------------------------------------+

# __surfactant_name__
#           valid phython disctionary label
# __denisty_1__
#           surfactant density in g/cm3
#           defaults to 1.0 g/cm3

# +--- frame 4: make stock -----------------------------------------------------+
# \                                                                             \
# \  concentration type {m/v, m/m}                                              \
# \  concentration: [__conc_1__]                                                \
# \  total mass: [__mass_total__] g                                             \
# \  ...display selected surfactant...                                          \
# \  ...display selected oil...                                                 \
# \                                                                             \
# \  +----------text_field-+                                                    \
# \  \                     \         mass_surf_actual: [__mass_surf__]          \
# \  \    recipe details   \         mass_oil_actual: [__mass_oil__]            \
# \  \                     \                                                    \
# \  +---------------------+                                                    \
# \                                                                             \
# +-----------------------------------------------------------------------------+

# +--- frame 5: measure density ------------------------------------------------+
# \                                                                             \
# \  label: [a substance for which to do that]                                  \
# \  volume : [__volume_1__]                                                    \
# \  mass : [__mass_ro__]                                                       \
# \                                                                             \
# +-----------------------------------------------------------------------------+



params = {
            "surfactant_label": "test_surf_aaa",
            "folder_name": "test_surf_aaa_fldr",
            "surfactant_pretty_name": "a test surf AAA",
            "ro": 0.989
            }


template_source = "//huckdfs-srv.science.ru.nl/huckdfs/RobotLab/Storage-Miscellaneous/aigars/temp/scripts/template"
root = "C:/Users/admin/Documents/Data/aikars/opentron"

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

project_root = os.path.join(root, params["folder_name"]) 
project_config = os.path.join(project_root, "config")
project_scripts = os.path.join(project_root, "scripts")
blank_config_path = os.path.join(project_config, "config_hld_ift_experiment__blank__opentron_pp.json") 
cmd_prompt_file = os.path.join(project_root, files_to_copy[0])
scan_settings_json = os.path.join(project_root, files_to_copy[1])
recipes_json = os.path.join(project_root, files_to_copy[2])
log_path = os.path.join(project_root, "log.log")
solution_repository_path = os.path.join(project_config, "solution_repository.json")

def modify_blank_profile(log_path):
    with open(blank_config_path, "r") as f:
        data = json.load(f)

    data["opentrons_api"]["log_path"] = log_path

    with open(blank_config_path, "w") as f:
        json.dump(data, f, indent = 4)

def copy_file(file_relative_path):
    shutil.copy(os.path.join(template_source, file_relative_path), os.path.join(project_root, file_relative_path))

def modify_command_prompt_bits():
    with open(cmd_prompt_file, "r") as fi:
        content = fi.read()

    lines = content.split("\n")

    lines[0] = f"cd {project_root}".replace("/", "\\")
    
    for i, l in enumerate(lines):
        lines[i] = lines[i] + "\n"

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

    data["recepies"] = list(map(make_recipe, ["heptane", "hexadecane"]))
    
    with open(recipes_json, "w") as fo:
        json.dump(data, fo, indent = 4)
       

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
        
    data["c_surfactant_experiment"] = def_conc_exp
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
        json.dump(data, fo, indent = 4)



print(project_root) # = os.path.join(root, folder_name) 
print(project_config) # = os.path.join(project_root, "config")
print(project_scripts) # = os.path.join(project_root, "scripts")
print(blank_config_path) # = os.path.join(project_config, "config_hld_ift_experiment__blank__opentron_pp.json") 
print(cmd_prompt_file) # = os.path.join(project_root, files_to_copy[0])
print(scan_settings_json) # = os.path.join(project_root, files_to_copy[1])
print(recipes_json) # = os.path.join(project_root, files_to_copy[2])
print(log_path) # = os.path.join(project_root, "log.log")
print(solution_repository_path) # = os.path.join(project_config, "solution_repository.json"

os.makedirs(project_root)
os.makedirs(project_config)
os.makedirs(project_scripts)

for f in files_to_copy:
    copy_file(f)

modify_blank_profile(log_path)
modify_command_prompt_bits()
create_recipes()
create_scan_settings()
