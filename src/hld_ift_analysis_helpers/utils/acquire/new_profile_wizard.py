
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




#-=====================================================================
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
        self.params["solution_repository"] = self.sol_rep_entry.get()
        self.params["settings_path"] = self.settings_entry.get()
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
        
        self.target_label = tk.Label(
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
        self.copy_verbose(self.params["solution_repository"], self.sol_rep_path)

    def settings_copy(self): 
        self.copy_verbose(self.params["settings_path"], self.settings_path)

    def get_blank_profile_name(self):
        with open(self.settings_path) as f:
            cfg = json.load(f)
        return cfg["configurations"]["blank"]

    def list_blank_config_files(self):
        all_files = os.listdir(os.path.join(self.source, "config"))
        search_for = [self.get_blank_profile_name()]
        return [st for st in all_files if any(sub in st for sub in search_for)] 

################################################################################

params = dict(
                source = "",
                solution_repository = "",
                settings_path = "",
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

# need to:
#   make folder
#   copy config/ blank configs
#   copy sol rep
#   command prompt bits
#   settings

