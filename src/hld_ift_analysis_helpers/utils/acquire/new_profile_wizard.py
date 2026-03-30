
STATE_LOCATION_SPECIFIED = 1

state = 0
current_params = dict()

def is_exit_signal(an_input: str):
    return an_input == "exit"

def exit_routine():
    print("Wizard will be terminated!!!")
    sys.exit()

if state < STATE_LOCATION_SPECIFIED:
    done = False
    while not done:
        source_path = input("Which folder to use as a template for new experiment?")
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

