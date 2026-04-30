#========================================================== notes
# select 
#   source folder, solution rep, scan_settings
# select
#   target folder
# create target folder, copy solution rep, config templates,
#           scan settings, command_prompt bits
#
# edit scan settings
#================================================================

import sys
import os
import tkinter as tk
from tkinter.filedialog import askopenfilename, askdirectory

STATE_LOCATION_SPECIFIED = 1

state = 0
current_params = dict()

def is_exit_signal(an_input: str):
    return an_input == "exit"

def exit_routine(an_input, params):
    print("Wizard will be terminated!!!")
    sys.exit()


def looping_input(a_dict, prompt, params):
    done = False
    while not done:
        an_input = input(prompt)

        if "main" not in a_dict.keys():
            print("cannot live without `main`")

        for a_key in a_dict.keys():
            if a_key == "main":
                done = a_dict["main"](an_input, params)
            elif a_dict[a_key]["value"] == an_input:
                done = a_dict[a_key]["fn"](an_input, params)


def is_valid_source_folder(an_input, params):
    status = os.path.exists(an_input) and os.path.isdir(an_input)
    if status:
        params["source_folder"] = an_input
    return status

def source_folder_dialog(an_input, params):
    folder = tk.filedialog.askdirectory(title='Select source folder...', mustexist = True)
    status = is_valid_source_folder(folder, params)
    return status
    
select_source_folder = dict(
                        main = is_valid_source_folder, 
                        exit = dict(value = "exit", fn = exit_routine), 
                        dlg  = dict(value = "dlg", fn = source_folder_dialog)
                        )
looping_input(
            select_source_folder, 
            "Which folder to use as a template for new experiment?\n   type exact location,\n   `dlg` to start dialog, OR\n   `exit`\n>>>",
            current_params
            )




print(current_params)
print("---------------------------------------------------")
exit()

if state < STATE_LOCATION_SPECIFIED:
    done = False
    while not done:
        source_path = input("Which folder to use as a template for new experiment?\n   type exact location OR\n   type `dlg` to start dialog\n>>>")
        if is_exit_signal(source_path):
            exit_routine()        



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


#-===================================================================== DETAILS STOCK CHOICE
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


# bits needed:
#   edit solution repository (add surfactant)
#   list solutions ==> make menuchoice
#   use previous oils and water bits as 1st choice 
#   list components in solutions

#====================================================================== UPDATE PATH VALUES

# open scan settings
# 
#  "DATA_PATH": main_folder
#  "LOG_PATH": "main_folder/og.log
#  "CONFIG_PATH": main_folder/config",
#  "SOLUTION_REPOSITORY_PATH": main_folder/config/solution_repository.json",

#================================================================================

#================================================================================
#examples:
#    open image:
#        https://stackoverflow.com/questions/10133856/how-to-add-an-image-in-tkinter
#   

# Do you want to use existing solution as recipe for stock?
#   yes/no

# on yes:
# select solution to use as template for stock solution
#   use chosen solution to make a recipe
#           ==> recipe 1

# on no:
# if surfactant no selected
#   select surfactant to use
#       if new_surfactant... selected
#           make new surfactant
#       else:
#           use chosen surfactant
# select oil
#       if new_oil... selected
#           make new oil
#       else:
#           use chosen oil
# what concentration to use?

# make recipe based on:
#       surfactant
#       oil
#       concentration

# INSTRUCTIONS
#       make solutions according to recipes
#
#       to measure density
#           take an aliquot (e.g. 5 mL) of reference fluid (water) and measure its mass
#           for each solution made:
#               take an aliquot (same volume as referece fluid) and measure its mass
#       continue with current script

# recipe x
# name of solution: enter string
#
# enter a mass used for each components:
# 
# density
#   refence fluid mass
#   solution mass
# ==> solution definition x

#================================================================================


from hld_ift_http.solution import Solution, Solution_Component
from hld_ift_http.solution_repository import Solution_Repository
from scipy.optimize import minimize
import numpy as np
import datetime
import json

#fro hld_ift_analysis_helpers.utils.acquire.generate_recepies_for_solutions_v2 import create_recepie_for


#> def create_recipe_for_user(surfactant_name, solution_name, concentration, sol_rep, quantity, quantity_type):
#>     surfactant = sol_rep.items[surfactant_name]
#>     solution = sol_rep.items[solution_name]
#> 
#>     if valid surfactant:
#>         if surfactant not in solution:
#>             binary mix of surfactant and solution at concentration
#>         else 
#>             binary mix of surfactant and rest of components at given proportions to yield surfactant at concentration
#>     else:
#>         recipe of solution
#>     
#>     how much: quantity / sum(components)
#>     sum(components) / density ==> vx
#>     v/vx = mult
#> 
#>     v * density / sum(components)

#> def create_recipe_from_solution_definition(solution, quantity, quantity_type):
#>     
#>     total_mass = quantity * solution.ro if quantity_type == "ml" else quantity
#> 
#>     recipe = f'to make {solution.name}`:\n'
#> 
#>     for component in solution.components:
#>         mass_comp = component.w * total_mass
#>         recipe += f'    {mass_comp:0.4f} g of `{component.label}`\n'
#> 
#>     return recipe


#> def create_solution_from_ingredients(lst_ingredients, name, m_ref, m_sol, sol_rep)
#>     ingredients_to_use = list(map(lambda x: dict(mass = x["mass"], solution = sol_rep.item(x["name"])), lst_ingredients))
#>     sol = list(functools.reduce(lambda x,y: mix(x["solution"], x["mass"], y["solution"], y["mass"], 1.0, "__"), ingredients_to_use))
#>     sol.ro = m_sol/m_ref
#>     sol.name = name
#>     return sol


def generate_recipe_pretty(sol_rep, stock_lbl, surfactant = ""):

    def choice_existing_solution_to_recipe(label):
        chosen = False
        while not chosen:
            k = input(f"use existing solution for `{label}` stock recipe? (Yy/Nn)")
            choice = k.upper()
            if choice == "Y":
                print("will use existing solution to make a recipe")
                chosen = True
            else: 
                print(f"will make stock from surfactant and a chosen `{label}`")
                chosen = True
        return choice == "Y"

    def create_new_surfactant(name_use, ro):
        #k = input("surfactant name:")
        #ro = float(input("surfactant density, g/ml:"))

        #name_use = k
        component = Solution_Component(name_use, 1)
        a_sol = Solution(name_use, ro, name_use, {})
        a_sol.add_component(component)
        return a_sol

    def print_lot_of_choices(choices, n = 30):
        for i,ch in enumerate(choices):
            print(ch)
            if i % n == n - 1:
                input("press enter to continue")

    def new_surfactant():
        names_taken = sol_rep.list_solution_names()

        chosen_name = False
        chosen_density = False

        name_use = ""
        ro = 1.0

        while not chosen_name:
            k = input("select surfactant name:")
            if k in names_taken:
                print("this name is already taken, pick different one!!!")
            else:
                chosen_name = True
                name_use = k

        while not chosen_density:
            k = input("select density, g/ml:")
            try:
                ro = float(k)
                chosen_density = True
            except:
                k = input("would you like to use 1.0 g/ml as default value? (Y/N)")
                if k.upper() == "Y":
                    ro = 1.0
                    chosen_density = True
        xxx = create_new_surfactant(name_use, ro)
        print("2222222222222222222222222222222222")
        print(type(xxx))
        print(str(xxx))
        print("2222222222222222222222222222222222")
        return xxx
                    
                      
    def select_surfactant():
        components = sol_rep.list_components()
        choices = ["new surfactant ..."] + components
        choices_pretty = [ f'{i: 4}: {v}' for i,v in enumerate(choices) ]
        chosen = False
        while not chosen:
            print_lot_of_choices(choices_pretty, n = 30) 
            k = input("select surfactant (or 0 to create a new one)")
            try:
                choice = int(k)
                if choice >= 0 or choice < len(choices_pretty):
                    chosen = True
            except:
                pass

        surfactant = None

        if choice == 0:
            surfactant = new_surfactant()    
            print("----------------------------")
            print(type(surfactant))
            print(str(surfactant))
            print("----------------------------")
            # add surfactant to sol_rep
            sol_rep.add_solution(surfactant)
            return surfactant
        else:
            return sol_rep.items[choices[choice]]

    def quantity_needed():
        chosen_quantity = False
        chosen_type = False

        default_quantity = 50
        default_type = "g"

        #while not chosen_quantity:
        k = input("quantity of solution required? (50)")
        try:
            quantity = float(k)
            chosen_quantity = True
        except:
            quantity = default_quantity
            chosen_quantity = True

        #while not chosen_type:
        k = input("0: mass, g OR 1: volume, ml?")
        try:
            choice = int(k)
            if choice == 1:
                return (quantity, "ml")
            else:
                return (quantity, "g")
        except:
            return (quantity, "g")
    
    def pick_existing_solution():
        all_names = sol_rep.list_solution_names()
        choices_pretty = [ f'{i: 4}: {v}' for i,v in enumerate(all_names) ]

        chosen = False
        while not chosen:
            print_lot_of_choices(choices_pretty, n = 30) 
            k = input("select solution to use for recipe:")
            try:
                choice = int(k)
                if choice >= 0 or choice < len(choices_pretty):
                    chosen = True
            except:
                pass

        return sol_rep.items[all_names[choice]]

    def target_concentration():
        print("can specify target concentration as `m/v` or `m/m`")
        k = input("0: m/v (default); 1: m/m")
        try:
            choice_conc = int(k)
            if choice_conc != 1:
                choice_conc = 0
        except:
            choice_conc = 0
        k = input("enter surfactant target concentration: (range ~ [0; 0.2]):")
        try:
            target_conc = float(k)
            if target_conc < 0 or target_conc > 0.25:
                print(f'target concentration ({target_conc}) outside of reasonable range, will use 0.2 instead!!!')
                target_conc = 0.2
        except:
            target_conc = 0.2
        return (target_conc, ['m/v', 'm/m'][choice_conc])
         
    def create_recipe_from_solution_definition(solution, quantity, quantity_type):
        
        total_mass = quantity * solution.ro if quantity_type == "ml" else quantity
    
        recipe = f'to make {solution.name}`:\n'
    
        for lbl in solution.components.keys():
            component = solution.components[lbl]
            mass_comp = component.w * total_mass
            recipe += f'    {mass_comp:0.4f} g of `{component.label}`\n'
    
        print("###############################")
        print(recipe)
        print("###############################")

        return recipe


    x = choice_existing_solution_to_recipe(stock_lbl)

    quantity = quantity_needed()
    recipe_dict = None

    if x:
        # recipe from existing
        solution = pick_existing_solution()
        recipe_dict = dict(
                        recipe = create_recipe_from_solution_definition(solution, quantity[0], quantity[1]), #human readable bit
                        components = solution.components.keys()
                        )
    else:
        # surf+oil
        surfactant_obj = None
        surfactant_out = ""
        if surfactant in sol_rep.list_solution_names():
            # use surfactant
            surfactant_obj = sol_rep.items[params["surfactant"]]
        else:
            surfactant_obj = select_surfactant()
            surfactant_out = surfactant_obj.name

        solvent = sol_rep.items[stock_lbl]

        target_conc = target_concentration() 

        recipe_dict = dict(
                        recipe = create_recepie_for(
                                            surfactant_obj,
                                            solvent,
                                            surfactant_obj.name,
                                            target_conc[0],
                                            target_conc[1],
                                            quantity[0],
                                            method = "Nelder-Mead"
                                            ),
                        component = [ surfactant_obj.name, solvent.name ]
                        )
        
    return recipe_dict


#=====================================================================
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
        f'    \"date\":       \"{datetime.datetime.now().strftime("%Y-%m-%d")}\"\n' + \
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


#=====================================================================

   
SOLUTION_REPOSITORY_PATH = '//huckdfs-srv.science.ru.nl/huckdfs/RobotLab/Storage-Miscellaneous/aigars/temp/HLD_scan/test_001/config/solution_repository.json'

rep = Solution_Repository.fromJSON(file = SOLUTION_REPOSITORY_PATH)

xyz = generate_recipe_pretty(rep, "heptane", "")

print(xyz)




#================================================================================

   

import os
import shutil
import json
import tkinter as tk
from tkinter.filedialog import askopenfilename, askdirectory
#from Tkinter import *


class DialogSource:

    def __init__(self, params: dict):
        self.params = params
        self.root = tk.Tk()
        self.root.geometry("550x300+300+150")

        self.source_var=tk.StringVar()
        self.sol_rep_var=tk.StringVar()
        self.settings_var=tk.StringVar()
        
        self.frame = tk.Frame(self.root)
        self.frame.pack()
        
        self.placeholder_1 = tk.Label(self.frame, text = '', font=('calibre',10, 'bold'))
        self.placeholder_2 = tk.Label(self.frame, text = '', font=('calibre',10, 'bold'))
        self.placeholder_3 = tk.Label(self.frame, text = '', font=('calibre',10, 'bold'))
        
        self.button = tk.Button (self.frame, text = "Next >>>", command = self.close_window)
        
        self.source_label = tk.Label(
                                self.frame,
                                text = 'source',
                                font=('calibre',10, 'bold')
                                )
        self.source_entry = tk.Entry(
                                self.frame,
                                textvariable = self.source_var,
                                font=('calibre',10,'normal'),
                                width = 50
                                )
        self.source_button = tk.Button(
                                self.frame,
                                text = "...",
                                command = self.choose_source
                                )
        
        self.sol_rep_state = tk.BooleanVar()
        
        self.sol_rep_non_default = tk.Checkbutton(
                                            self.frame,
                                            text='default location', 
                                            command=self.metricChanged,
                                            variable=self.sol_rep_state,
                                            onvalue='metric',
                                            offvalue='imperial',
                                            state = tk.DISABLED
                                            )
        #sol_rep_non_default.select()
        
        self.sol_rep_label = tk.Label(
                                self.frame,
                                text = 'solution repository',
                                font=('calibre',10, 'bold')
                                )
        self.sol_rep_entry = tk.Entry(
                                    self.frame,
                                    textvariable = self.sol_rep_var,
                                    font=('calibre',10,'normal'),
                                    width = 50
                                    )
        self.sol_rep_button = tk.Button(
                                    self.frame,
                                    text = "...",
                                    command = self.choose_sol_rep
                                    )

        self.settings_label = tk.Label(
                                self.frame,
                                text = 'scan settings',
                                font=('calibre',10, 'bold')
                                )
        self.settings_entry = tk.Entry(
                                    self.frame,
                                    textvariable = self.settings_var,
                                    font=('calibre',10,'normal'),
                                    width = 50
                                    )
        self.settings_button = tk.Button(
                                    self.frame,
                                    text = "...",
                                    command = self.choose_settings
                                    )
        
        #button.pack()
        
        source_row = 0
        sol_rep_row_1 = 2
        sol_rep_row_2 = sol_rep_row_1 + 1
        settings_row = 4 
        self.source_label.grid(row= source_row,column=0)
        self.source_entry.grid(row= source_row,column=1)
        self.source_button.grid(row= source_row,column=2)
        self.placeholder_1.grid(row=1,column=0)
        self.sol_rep_non_default.grid(row=sol_rep_row_1,column=1)
        self.sol_rep_label.grid(row=sol_rep_row_2,column=0)
        self.sol_rep_entry.grid(row=sol_rep_row_2,column=1)
        self.sol_rep_button.grid(row=sol_rep_row_2,column=2)
        self.settings_label.grid(row=settings_row,column=0)
        self.settings_entry.grid(row=settings_row,column=1)
        self.settings_button.grid(row=settings_row,column=2)
        self.placeholder_2.grid(row=4,column=0)
        self.placeholder_3.grid(row=5,column=0)
        self.button.grid(row=6,column=2)
        
        self.root.mainloop()

    def close_window(self): 
        self.params["source"] = self.source_entry.get()
        self.params["source_solution_repository"] = self.sol_rep_entry.get()
        self.params["source_settings_path"] = self.settings_entry.get()
        self.root.destroy()

    def openfn(self):
        filename = tk.filedialog.askopenfilename(title='open')
        return filename
    
    def opendir(self):
        folder = tk.filedialog.askdirectory(title='open')
        return folder
    
    def choose_source(self):
        out = self.opendir()    
        self.source_entry.delete(0, tk.END) #deletes the current value
        self.source_entry.insert(0, out) #inserts new value assigned by 2nd parameter
        self.update_sol_rep_entry(self.default_sol_rep(out))
        self.update_settings(self.default_settings(out))
    
    def choose_sol_rep(self):
        out = self.openfn()
        self.update_sol_rep_entry(out)
    
    def metricChanged(self):
        return 1
    
    def default_sol_rep(self, folder):
        if os.path.exists(folder) and os.path.isdir(folder):
            def_path = os.path.join(folder, "config/solution_repository.json")
            if os.path.exists(def_path):
                return def_path
        return ""
    
    def update_sol_rep_entry(self, value):
        self.sol_rep_entry.delete(0, tk.END) #deletes the current value
        self.sol_rep_entry.insert(0, value) #inserts new value assigned by 2nd parameter
    
    def choose_settings(self):
        out = self.openfn()
        self.update_settings(out)
    
    def default_settings(self, folder):
        if os.path.exists(folder) and os.path.isdir(folder):
            def_path = os.path.join(folder, "scan_settings.json")
            if os.path.exists(def_path):
                return def_path
        return ""
    
    def update_settings(self, value):
        self.settings_entry.delete(0, tk.END) #deletes the current value
        self.settings_entry.insert(0, value) #inserts new value assigned by 2nd parameter
    


class DialogTarget:
    def __init__(self, params: dict):
        self.params = params
        self.root = tk.Tk()
        self.root.geometry("550x300+300+150")

        self.target_var=tk.StringVar()
        self.folder_var=tk.StringVar()
        
        self.frame = tk.Frame(self.root)
        self.frame.pack()
        
        self.placeholder_1 = tk.Label(self.frame, text = '', font=('calibre',10, 'bold'))
        self.placeholder_2 = tk.Label(self.frame, text = '', font=('calibre',10, 'bold'))
        self.placeholder_3 = tk.Label(self.frame, text = '', font=('calibre',10, 'bold'))
        
        self.bt_next = tk.Button (self.frame, text = "Next >>>", command = self.close_window)
        self.bt_back = tk.Button (self.frame, text = "<<< Back", command = self.close_window)
        
        self.target_label= tk.Label(
                                self.frame,
                                text = 'target folder',
                                font=('calibre',10, 'bold')
                                )
        self.target_entry = tk.Entry(
                                self.frame,
                                textvariable = self.target_var,
                                font=('calibre',10,'normal'),
                                width = 50
                                )
        self.target_button = tk.Button(
                                self.frame,
                                text = "...",
                                command = self.choose_target
                                )

        self.folder_label = tk.Label(
                                self.frame,
                                text = 'new folder',
                                font=('calibre',10, 'bold')
                                )
        self.folder_entry = tk.Entry(
                                self.frame,
                                textvariable = self.folder_var,
                                font=('calibre',10,'normal'),
                                width = 50
                                )
        #>self.folder_button = tk.Button(
        #>                        self.frame,
        #>                        text = "...",
        #>                        command = self.choose_source
        #>                        )
        
        #> self.sol_rep_state = tk.BooleanVar()
        #> 
        #> self.sol_rep_non_default = tk.Checkbutton(
        #>                                     self.frame,
        #>                                     text='default location', 
        #>                                     command=self.metricChanged,
        #>                                     variable=self.sol_rep_state,
        #>                                     onvalue='metric',
        #>                                     offvalue='imperial',
        #>                                     state = tk.DISABLED
        #>                                     )
        #sol_rep_non_default.select()
        
        #> self.sol_rep_label = tk.Label(
        #>                         self.frame,
        #>                         text = 'solution repository',
        #>                         font=('calibre',10, 'bold')
        #>                         )
        #> self.sol_rep_entry = tk.Entry(
        #>                             self.frame,
        #>                             textvariable = self.sol_rep_var,
        #>                             font=('calibre',10,'normal'),
        #>                             width = 50
        #>                             )
        #> self.sol_rep_button = tk.Button(
        #>                             self.frame,
        #>                             text = "...",
        #>                             command = self.choose_sol_rep
        #>                             )
        
        
        target_row = 0
        folder_row = 2
        
        self.target_label.grid(row= target_row,column=0)
        self.target_entry.grid(row= target_row,column=1)
        self.target_button.grid(row= target_row,column=2)
        self.placeholder_1.grid(row=1,column=0)
        self.folder_label.grid(row= folder_row,column=0)
        self.folder_entry.grid(row= folder_row,column=1)
        self.placeholder_2.grid(row=4,column=0)
        self.placeholder_3.grid(row=5,column=0)
        self.bt_back.grid(row=6,column=1)
        self.bt_next.grid(row=6,column=2)
        
        self.root.mainloop()

    def close_window(self): 
        self.params["target"] = self.target_entry.get()
        self.params["folder"] = self.folder_entry.get()
        self.root.destroy()

    def openfn(self):
        filename = tk.filedialog.askopenfilename(title='open')
        return filename
    
    def opendir(self):
        folder = tk.filedialog.askdirectory(title='open')
        return folder
    
    def choose_target(self):
        out = self.opendir()    
        self.target_entry.delete(0, tk.END) #deletes the current value
        self.target_entry.insert(0, out) #inserts new value assigned by 2nd parameter
    
    def metricChanged(self):
        return 1
    
    
class CreateNewExperimentSet:
    def __init__(self, params: dict):
        self.params = params
        self.source = params["source"]
        self.make_folders()
        self.do_copy()
        self.sol_rep_copy()
        self.settings_copy()
        print("----------------")
        print(self.list_blank_config_files())
        for fl in self.list_blank_config_files():
            self.copy_verbose(
                    os.path.join(self.source, "config", fl),
                    os.path.join(self.config_folder, fl)
                    )
        print("----------------")
        self.edit_settings_pathes()
        self.params["scan_settings_path"] = self.settings_path
        #self.new_surfactant_add()

    def make_folders(self):
        self.main_folder = os.path.join(self.params["target"], self.params["folder"])
        self.config_folder = os.path.join(self.main_folder, "config") 
        self.sol_rep_path = os.path.join(self.config_folder, "solution_repository.json")
        self.settings_path = os.path.join(self.main_folder, "scan_settings.json")

        if os.path.exists(self.main_folder):
            print(f'!!! ERROR !!!\n   {self.main_folder}\nalready exists!!! Choose folder name that does not point to existing folder.\n ... will stop now ...')
            exit()

        
        os.mkdir(self.main_folder)
        print(f'making folder:\n    {self.main_folder}')
        os.mkdir(self.config_folder)
        print(f'making folder:\n    {self.config_folder}')

    def copy_verbose(self, p1, p2):
        print(f'copy\n   {p1}\n    ==>\n    {p2}')
        shutil.copy(p1, p2)

    def copy_file_relative(self, file_relative_path):
        self.copy_verbose(os.path.join(self.source, file_relative_path), os.path.join(self.main_folder, file_relative_path))

    def do_copy(self):
        files_to_copy = [
                    "command_prompt_bits.md", 
                    #"scan_settings.json",
                    "recipes_solutions.json"
                    #"config/solution_repository.json"
                    #"config/config_hld_ift_experiment__blank__opentron_pp.json",
                    #"config/config_hld_ift_experiment__blank__execute_measurement.json",
                    #"config/config_hld_ift_experiment__blank__execute_measurement__no_wash.json",
                    #"config/config_hld_ift_experiment__blank__execute_measurement__wash.json",
                    #"config/config_hld_ift_experiment__blank__mixing_graph.json"
                    ]
        for f in files_to_copy:
            self.copy_file_relative(f)

    def sol_rep_copy(self): 
        self.copy_verbose(self.params["source_solution_repository"], self.sol_rep_path)

    def settings_copy(self): 
        self.copy_verbose(self.params["source_settings_path"], self.settings_path)

    def get_blank_profile_name(self):
        with open(self.settings_path) as f:
            cfg = json.load(f)
        return cfg["configurations"]["blank"]

    def list_blank_config_files(self):
        all_files = os.listdir(os.path.join(self.source, "config"))
        search_for = [self.get_blank_profile_name()]
        return [st for st in all_files if any(sub in st for sub in search_for)] 
    def new_surfactant_add(self):
        with open(self.sol_rep_path, "r") as f:
            cfg = json.load(f)
        print(json.dumps(cfg, indent=2))

        sol_rep = Solution_Repository.fromJSON(cfg)

        AddNewSurfactantDialog(sol_rep)

        print(sol_rep.toJSON(indent = 2))
    def edit_settings_pathes(self):
        with open(self.settings_path, "r") as f:
            cfg = json.load(f)

        cfg["DATA_PATH"] = self.main_folder
        cfg["LOG_PATH"] = os.path.join(self.main_folder, "log.log")
        cfg["CONFIG_PATH"] = self.config_folder
        cfg["SOLUTION_REPOSITORY_PATH"] = self.sol_rep_path

        with open(self.settings_path, "w") as f:
            json.dump(cfg, f, indent = 2)


################################################################################
# new_surfactant_dlg.py!!!
# gui for new surfactant addition
import tkinter as tk
from hld_ift_http.solution_repository import Solution_Repository

class AddNewSurfactantDialog:
    def __init__(self, sol_rep: Solution_Repository):
        self.sol_rep = sol_rep
        self.root = tk.Tk()
        self.root.geometry("550x300+300+150")

        self.surfactant_name=tk.StringVar()
        self.surfactant_density=tk.StringVar()


        self.frame = tk.Frame(self.root)
        self.frame.pack()
        
        self.placeholder_1 = tk.Label(self.frame, text = '', font=('calibre',10, 'bold'))
        self.placeholder_2 = tk.Label(self.frame, text = '', font=('calibre',10, 'bold'))
        
        self.bt_OK = tk.Button (self.frame, text = "OK", command = self.update_and_close_window)
        self.bt_Cancel = tk.Button (self.frame, text = "Cancel", command = self.close_window)
        
        self.surfactant_label = tk.Label(
                                self.frame,
                                text = 'name',
                                font=('calibre',10, 'bold')
                                )
        self.surfactant_entry = tk.Entry(
                                self.frame,
                                textvariable = self.surfactant_name,
                                font=('calibre',10,'normal'),
                                width = 50
                                )
       
        self.surfactant_density_label = tk.Label(
                                self.frame,
                                text = 'density, g/mL',
                                font=('calibre',10, 'bold')
                                )
        self.surfactant_density_entry = tk.Entry(
                                    self.frame,
                                    textvariable = self.surfactant_density,
                                    text = "1.0",
                                    font=('calibre',10,'normal'),
                                    width = 50
                                    )
       
        source_row = 0
        self.surfactant_label.grid(row= source_row,column=0)
        self.surfactant_entry.grid(row= source_row,column=1)
        self.placeholder_1.grid(row=1,column=0)
        self.surfactant_density_label.grid(row=2,column=0)
        self.surfactant_density_entry.grid(row=2,column=1)
        self.placeholder_2.grid(row=3,column=0)
        self.bt_OK.grid(row=4,column=1)
        self.bt_Cancel.grid(row=4,column=2)
        
        self.root.mainloop()

    def close_window(self): 
        self.root.destroy()
    def update_and_close_window(self):
        ro = float(self.surfactant_density_entry.get())
        name = self.surfactant_entry.get()
        self.sol_rep.add_item(dict(
                                mixture_type = "pure_compound",
                                label = name,
                                name = name,
                                ro = ro
                                ))
        self.close_window()


################################################################################

class EditStockSolutions:
    def __init__(self, path_scan_settings, parent):
        self.path_scan_settings = path_scan_settings
        self.root = parent
        self.open_settings()
        self.open_sol_rep()

        #self.make_dlg()
        #self.root = tk.Tk()
        self.root.title("edit stock solutions")
        self.root.geometry("550x700")


        #>self.source_var=tk.StringVar()
        #>self.sol_rep_var=tk.StringVar()
        #>self.settings_var=tk.StringVar()
        
        #11111111>>> self.frame = tk.Frame(self.root)
        #11111111>>> self.frame.pack()
        #11111111>>> 
        #11111111>>> self.placeholder_1 = tk.Label(self.frame, text = '', font=('calibre',10, 'bold'))
        #11111111>>> self.placeholder_2 = tk.Label(self.frame, text = '', font=('calibre',10, 'bold'))
        #11111111>>> self.placeholder_3 = tk.Label(self.frame, text = '', font=('calibre',10, 'bold'))
        #11111111>>> self.placeholder_4 = tk.Label(self.frame, text = '', font=('calibre',10, 'bold'))
        #11111111>>> self.placeholder_5 = tk.Label(self.frame, text = '', font=('calibre',10, 'bold'))
        #11111111>>> 
        #11111111>>> self.bt_OK = tk.Button (self.frame, text = "OK", command = self.close_window)
        #11111111>>> 
        #11111111>>> self.surf_list = self.all_components()
        #11111111>>> self.surf_value = tk.StringVar(self.root)
        #11111111>>> self.surf_value.set("Select a surfactant")
        #11111111>>> self.surf_menu = tk.OptionMenu(self.frame, self.surf_value, *self.surf_list)
        #11111111>>> self.surfactant_label = tk.Label(
        #11111111>>>                         self.frame,
        #11111111>>>                         text = 'surfactant',
        #11111111>>>                         font=('calibre',10, 'bold')
        #11111111>>>                         )

        #11111111>>> self.bt_new_surf = tk.Button(
        #11111111>>>                             self.frame,
        #11111111>>>                             text = "new surfactant ...",
        #11111111>>>                             command = self.new_surfactant_add
        #11111111>>>                             )
        #11111111>>> 
        #11111111>>> self.sol_list = self.all_solutions()

        #11111111>>> self.oil_1_val = tk.StringVar(self.root) 
        #11111111>>> self.oil_1_val.set(self.settings["stocks"]["oil_1"])

        #11111111>>> self.oil_2_val = tk.StringVar(self.root) 
        #11111111>>> self.oil_2_val.set(self.settings["stocks"]["oil_2"])

        #11111111>>> self.aqu_1_val = tk.StringVar(self.root) 
        #11111111>>> self.aqu_1_val.set(self.settings["stocks"]["stock_aqueous_1"])

        #11111111>>> self.aqu_2_val = tk.StringVar(self.root) 
        #11111111>>> self.aqu_2_val.set(self.settings["stocks"]["stock_aqueous_2"])

        #11111111>>> self.oil_1_menu = tk.OptionMenu(self.frame, self.oil_1_val, *self.sol_list)
        #11111111>>> self.oil_2_menu = tk.OptionMenu(self.frame, self.oil_2_val, *self.sol_list)
        #11111111>>> self.aqu_1_menu = tk.OptionMenu(self.frame, self.aqu_1_val, *self.sol_list)
        #11111111>>> self.aqu_2_menu = tk.OptionMenu(self.frame, self.aqu_2_val, *self.sol_list)

        #11111111>>> self.oil_1_label = tk.Label(
        #11111111>>>                         self.frame,
        #11111111>>>                         text = 'oil 1',
        #11111111>>>                         font=('calibre',10, 'bold')
        #11111111>>>                         )
        #11111111>>> self.oil_2_label = tk.Label(
        #11111111>>>                         self.frame,
        #11111111>>>                         text = 'oil 2',
        #11111111>>>                         font=('calibre',10, 'bold')
        #11111111>>>                         )
        #11111111>>> self.aqu_1_label = tk.Label(
        #11111111>>>                         self.frame,
        #11111111>>>                         text = 'aqueous 1',
        #11111111>>>                         font=('calibre',10, 'bold')
        #11111111>>>                         )
        #11111111>>> self.aqu_2_label = tk.Label(
        #11111111>>>                         self.frame,
        #11111111>>>                         text = 'aqueous 2',
        #11111111>>>                         font=('calibre',10, 'bold')
        #11111111>>>                         )

        
        self.surfactant_name=tk.StringVar()
        self.surfactant_density=tk.StringVar()

        self.placeholder_1 = tk.Label(self.root, text = '', font=('calibre',10, 'bold'))
        self.placeholder_2 = tk.Label(self.root, text = '', font=('calibre',10, 'bold'))
        self.placeholder_3 = tk.Label(self.root, text = '', font=('calibre',10, 'bold'))
        self.placeholder_4 = tk.Label(self.root, text = '', font=('calibre',10, 'bold'))
        self.placeholder_5 = tk.Label(self.root, text = '', font=('calibre',10, 'bold'))
        self.placeholder_X = tk.Label(self.root, text = '', font=('calibre',10, 'bold'))
        
        self.bt_OK = tk.Button (self.root, text = "OK", command = self.close_window)
        
        self.surf_list = self.all_components()
        self.surf_value = tk.StringVar(self.root)
        self.surf_value.set("Select a surfactant")
        self.surf_menu = tk.OptionMenu(self.root, self.surf_value, *self.surf_list)
        self.new_surfactant_label = tk.Label(
                                self.root,
                                text = 'name',
                                font=('calibre',10, 'bold')
                                )
        self.new_surfactant_entry = tk.Entry(
                                self.root,
                                textvariable = self.surfactant_name,
                                font=('calibre',10,'normal'),
                                width = 50
                                )
       
        self.surfactant_density_label = tk.Label(
                                self.root,
                                text = 'density, g/mL',
                                font=('calibre',10, 'bold')
                                )
        self.surfactant_density_entry = tk.Entry(
                                    self.root,
                                    textvariable = self.surfactant_density,
                                    text = "1.0",
                                    font=('calibre',10,'normal'),
                                    width = 50
                                    )
 
        print("first instance calling")
        print(hex(id(self.surf_menu)))
        print(hex(id(self.surf_menu["menu"])))
        self.ref_to_menu = self.surf_menu["menu"]
        self.surfactant_label = tk.Label(
                                self.root,
                                text = 'surfactant',
                                font=('calibre',10, 'bold')
                                )

        self.bt_new_surf = tk.Button(
                                    self.root,
                                    text = "new surfactant ...",
                                    command = self.update_new_surfactant
                                    )
        
        self.sol_list = self.all_solutions()

        self.oil_1_val = tk.StringVar(self.root) 
        self.oil_1_val.set(self.settings["stocks"]["oil_1"])

        self.oil_2_val = tk.StringVar(self.root) 
        self.oil_2_val.set(self.settings["stocks"]["oil_2"])

        self.aqu_1_val = tk.StringVar(self.root) 
        self.aqu_1_val.set(self.settings["stocks"]["stock_aqueous_1"])

        self.aqu_2_val = tk.StringVar(self.root) 
        self.aqu_2_val.set(self.settings["stocks"]["stock_aqueous_2"])

        self.oil_1_menu = tk.OptionMenu(self.root, self.oil_1_val, *self.sol_list)
        self.oil_2_menu = tk.OptionMenu(self.root, self.oil_2_val, *self.sol_list)
        self.aqu_1_menu = tk.OptionMenu(self.root, self.aqu_1_val, *self.sol_list)
        self.aqu_2_menu = tk.OptionMenu(self.root, self.aqu_2_val, *self.sol_list)

        self.oil_1_label = tk.Label(
                                self.root,
                                text = 'oil 1',
                                font=('calibre',10, 'bold')
                                )
        self.oil_2_label = tk.Label(
                                self.root,
                                text = 'oil 2',
                                font=('calibre',10, 'bold')
                                )
        self.aqu_1_label = tk.Label(
                                self.root,
                                text = 'aqueous 1',
                                font=('calibre',10, 'bold')
                                )
        self.aqu_2_label = tk.Label(
                                self.root,
                                text = 'aqueous 2',
                                font=('calibre',10, 'bold')
                                )

        #------------------------------
        
        
        self.surfactant_label.grid(row=0 ,column=0)
        self.surf_menu.grid(row=1 ,column=0)
        self.placeholder_X.grid(row = 1, column = 1)
        self.bt_new_surf.grid(row=1,column=2)
        self.new_surfactant_label.grid(row=1, column = 3)
        self.new_surfactant_entry.grid(row = 1, column = 4)
        self.surfactant_density_label.grid(row = 2, column = 3)
        self.surfactant_density_entry.grid(row = 2, column = 4)

        self.placeholder_1.grid(row=2, column=0)
        self.oil_1_label.grid(row=3, column=0)
        self.oil_1_menu.grid(row=4, column=0)

        self.placeholder_2.grid(row=5, column=0)
        self.oil_2_label.grid(row=6, column=0)
        self.oil_2_menu.grid(row=7, column=0)

        self.placeholder_3.grid(row=8, column=0)
        self.aqu_1_label.grid(row=9, column=0)
        self.aqu_1_menu.grid(row=10, column=0)

        self.placeholder_4.grid(row=11, column=0)
        self.aqu_2_label.grid(row=12, column=0)
        self.aqu_2_menu.grid(row=13, column=0)


        self.placeholder_5.grid(row=14, column=0)
        self.bt_OK.grid(row=15,column=2)

        #self.update_all_menus("how about it?")
        #self.root.mainloop()


    def update_new_surfactant(self):
        ro_str = self.surfactant_density_entry.get()
        ro = float("1.0" if ro_str == "" else ro_str)
        name = self.new_surfactant_entry.get()
        self.sol_rep.add_item(dict(
                                mixture_type = "pure_compound",
                                label = name,
                                name = name,
                                ro = ro
                                ))
        self.update_all_menus(name)
        self.new_surfactant_entry.delete(0, "end")
        self.surfactant_density_entry.delete(0, "end")

    def open_settings(self):
        with open(self.path_scan_settings,"r") as f:
            cfg = json.load(f)

        self.settings = cfg
    def write_settings(self):
        with open(self.path_scan_settings,"w") as f:
            json.dump(self.settings, f, indent = 2)

    def write_sol_rep(self):
        with open(self.settings["SOLUTION_REPOSITORY_PATH"],"w") as f:
            self.sol_rep.toJSON(file = f, indent = 2)

    def open_sol_rep(self):
        self.sol_rep = Solution_Repository.fromJSON(file = self.settings["SOLUTION_REPOSITORY_PATH"])

    def all_solutions(self):
        return self.sol_rep.list_solution_names()
        
    def all_components(self):
        return self.sol_rep.list_components()

    def update_settings(self):
        self.settings["stocks"]["oil_1"] = self.oil_1_val.get()
        self.settings["stocks"]["oil_2"] = self.oil_2_val.get()
        self.settings["stocks"]["stock_aqueous_1"] = self.aqu_1_val.get()
        self.settings["stocks"]["stock_aqueous_2"] = self.aqu_2_val.get()
        self.settings["stocks"]["surfactant"] = self.surf_value.get()

    def close_window(self):
        self.update_settings()
        self.write_sol_rep()
        self.write_settings()
        self.root.destroy()
    
    def update_menuoption_vals(self, a_menu, menu_var, all_options):
        menu = a_menu["menu"]
        menu.delete(0, "end")
        for string in all_options:
            menu.add_command(label=string, 
                             command=lambda value=string: menu_var.set(value))

    #def make_dlg(self):
    def new_surfactant_add(self):
        #print(self.sol_rep.toJSON(indent=2))
        print("#11111111111111111111111111111111111111111111111111111111111111111111111111111111")
        print(self.all_components())
        print(type(self.root))
        print(f'self.surf_menu id: {hex(id(self.surf_menu))}')
        print("#11111111111111111111111111111111111111111111111111111111111111111111111111111111")
        AddNewSurfactantDialog(self.sol_rep)
        self.update_all_menus()
        #print(self.sol_rep.toJSON(indent=2))
        print("#22222222222222222222222222222222222222222222222222222222222222222222222222222222")
        print(self.all_components())
        print("#22222222222222222222222222222222222222222222222222222222222222222222222222222222")

    def update_all_menus(self, an_opt: str = ""):
        print("a")
        for astr in self.all_solutions():
            if astr not in self.sol_list:
                self.sol_list.append(astr)

        print("b")
        for astr in self.all_components():
            if astr not in self.surf_list:
                self.surf_list.append(astr)

        print("c")
        print(hex(id(self.root)))
        print(self.root)
        #self.update_menuoption_vals(self.surf_menu, self.surf_value, surf_list)
        print(self.surf_menu)
        print(type(self.surf_menu))
        print(dir(self.surf_menu))
        #>...menu = self.surf_menu["menu"]
        #>...print(f'menu:\n{menu}')
        #>...print(type(menu))
        #>...print(dir(menu))
        print("d")
        #>...menu.delete(0, "end")
        print("e")
        #>...for st in self.surf_list:
        #>...    menu.add_command(label = st, command = lambda value = st: self.surf_value.set(value))
        print(type(self.root))
        print(hex(id(self.root)))
        print(hex(id(self.surf_menu)))
        print(hex(id(self.surf_menu["menu"])))
        new_str = an_opt if an_opt != "" else "to_add_now"
        #self.surf_menu["menu"].add_command(label = new_str, command = lambda value = new_str: self.surf_value.set(value))
        self.ref_to_menu.add_command(label = new_str, command = lambda value = new_str: self.surf_value.set(value))
        print("f")

        #> self.update_menuoption_vals(self.oil_1_menu, self.oil_1_val, sol_list)
        #> self.update_menuoption_vals(self.oil_2_menu, self.oil_2_val, sol_list)
        #> self.update_menuoption_vals(self.aqu_1_menu, self.aqu_1_val, sol_list)
        #> self.update_menuoption_vals(self.aqu_2_menu, self.aqu_2_val, sol_list)



#---------------------------------

#class ModifyScanSettings:

params = dict(
                source = "",
                source_solution_repository = "",
                source_settings_path = "",
                target = "",
                folder = "",
                value = 1,
                help = "none"
                ) 
print(params)
print("===================================================<<<<<<<<<<<<<<<<")
dw = DialogSource(params)

print(params)

dw2 = DialogTarget(params)

print(params)

dw3 = CreateNewExperimentSet(params)

print(params)

rroot = tk.Tk()
dw = EditStockSolutions(params["scan_settings_path"], rroot)
rroot.mainloop()

print(params)
# need to:
#   make folder
#   copy config/ blank configs
#   copy sol rep
#   command prompt bits
#   settings

