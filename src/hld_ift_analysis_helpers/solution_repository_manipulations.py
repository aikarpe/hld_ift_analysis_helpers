#___ # object to modify solution repository with user input
#___ 
#___ # open/save/close sol rep
#___ # list components/solutions
#___ # generate recipe
#___ # add substance/binary mixture
#___ 
#___ main_menu = ["Open/Save/Close repository", 
#___              "list items",
#___              "generate recipes",
#___              "add/remove items"
#___              ]
#___ 
#___ modify_rep_menu = ["Open", "Save", "Close"]
#___ list_menu = ["components", "mixtures/solutions", "inspect_item"]
#___ add_remove_menu = ["Add", "Remove"]
#___ 
#___ def open_user():
#___     if sol_rep is not None:
#___         close_user dlg
#___         if canceled: return 
#___         
#___         open_user dlg
#___         if canceled: return
#___         else:
#___             open selected
#___ 
#___ def close_user():
#___         # prompt saving
#___         ask_save/discard/cancel
#___         if wants to save
#___             save_user
#___             close
#___         elif wants to cancel operation
#___             return
#___         else
#___             close
#___ 
#___ def save_user():
#___         if sol_rep is not None
#___             save sol_rep
#___ 
#___ def list_components():
#___     k = input("filter>")
#___     a_list = get list
#___     apply k to a_list
#___     create menu with numbers
#___ 
#___ def list_solutions():
#___         get list
#___         print with numbers
#___ 
#___ def inspect_item():
#___         list_solutions()
#___         get selection
#___         pick selection
#___         show selection
#___ 
#___ def select_item() 
#___ 
#___ 
#___ # start program
#___ #   open repository
#___ #   make a copy of repository obj OR copy of its file (.bak)
#___ #   
#___ # main menu loop:
#___ #   make a choice
#___ #       if choice == exit
#___ #           break loop
#___ #       elif choice == add:
#___ #           add menu
#___ #       elif choice == delete:
#___ #           delete_menu
#___ #       elif choice == view:
#___ #           view_menu
#___ #       elif choice == recipe:
#___ #           recipe_menu
#___ #       elif choice == file_op:
#___ #           file_op_menu
#___ #
#___ # end program
#___ #   prompt: save changes y/n
#___ #   if y:
#___ #       save current state of sol rep
#___ #   else:
#___ #       restore initial from copy or bak
#___ 
#___ # add menu:
#___ #   make a choice
#___ #       if choice = add compound
#___ #       elif choice = add mixture
#___ 
#___ 
#___ # import item(s) from another repository
#___ # save changes
#___ # discard changes
#___ # exit
#___ 
#___ # delete item from repository
#___ # add new compound (1 component)
#___ # add new mixture
#___ # add new binary mixture:
#___ # recipe: existing mixture
#___ # recipe: from two mixtures
#___ 
#___ # menu_main
#___ #   import, delete, add, recipe, view

def looping_input(a_dict, prompt, params):
    done = False
    while not done:
        an_input = input(prompt)

        if "main" not in a_dict.keys():
            print("cannot live without `main`")

        processed = False

        for a_key in [x for x in a_dict.keys() if x != "main"]:
            if a_dict[a_key]["value"] == an_input:
                done = a_dict[a_key]["fn"](an_input, params)
                processed = True

        if not processed:
            done = a_dict["main"](an_input, params)


import functools
import datetime
import numpy as np
import json
from hld_ift_http.solution import Solution, Solution_Component
from hld_ift_http.solution_repository import Solution_Repository
from scipy.optimize import minimize

class SolutionRepositoryApp:
    """
        simple app for modification of solution repository
    """
    def __init__(self, path: str):
        self.rep = self.open(path) 
        self.original_path = path
        self.recipes = []
        #self.list_solutions()
        #self.list_components()
        #self.show_solution_details("nonsense")
        #self.show_solution_details_w_selection()
        #self.create_recipe_w_dialog()
        self.recipes = dict()
        self.main_dlg()

    def main_dlg(self):
        def print_title_box(a_str):
            print("================================================================================")
            print(f'|{a_str:^78}|')
            print("================================================================================")

        def save_rep_w_prompt():
            k = input(f'repository @:\n{self.original_path}\nDo you want to save it? y/n>>>')
            if k == "y":
                save_rep()
            elif k == "Y":
                save_rep()
            elif len(k) > 1 and k[0] in 'yY':
                save_rep()
        def save_rep():
            with open(self.original_path, 'w') as f:
                self.rep.toJSON(f, indent = 2)
        def is_unknown_choice(an_input, params):
            return False
        def exit_fn(an_input, params):
            save_rep_w_prompt()
            return True
        def or_else(an_input, params):
            k = input("... press enter to continue...")
            return False
        def user_recipe(an_input, params):
            print_title_box("Create recipe ...")
            self.create_recipe_w_dialog()
            k = input("... press enter to continue...")
            return False
        def user_list_solutions(an_input, params):
            print_title_box("List solutions ...")
            self.list_solutions()
            k = input("... press enter to continue...")
            return False
        def user_list_components(an_input, params):
            print_title_box("List available components ...")
            self.list_components()
            k = input("... press enter to continue...")
            return False
        def user_show_solution_details(an_input, params):
            print_title_box("Show details of a solution/mixture ...")
            self.show_solution_details_w_selection()
            k = input("... press enter to continue...")
            return False
        def user_add_cmpd(an_input, params):
            print_title_box("Add new compound ...")
            self.add_new_compound()
            k = input("... press enter to continue...")
            return False
        def user_list_recipes(an_input, params):
            print_title_box("List recipes ...")
            self.list_recipes()
            k = input("... press enter to continue...")
            return False
        def user_show_recipes_details(an_input, params):
            print_title_box("Show details of a recipe ...")
            self.show_recipe_details_w_selection()
            k = input("... press enter to continue...")
            return False
        def user_add_sol_from_recipe(an_input, params):
            print_title_box("Add solution created from recipe ...")
            self.add_solution_from_recipe()
            k = input("... press enter to continue...")
            return False

        mmparams = dict()
        select_main_menu = dict(
                        main = is_unknown_choice,
                        exit = dict(value = "exit", fn = exit_fn), 
                        cancel = dict(value = "cancel", fn = exit_fn), 
                        cancel2 = dict(value = "c", fn = exit_fn),
                        quit = dict(value = "q", fn = exit_fn),
                        recipe = dict(value = "r", fn = user_recipe),
                        list_sol = dict(value = "vs", fn = user_list_solutions),
                        list_cmp = dict(value = "vc", fn = user_list_components),
                        list_dtls = dict(value = "vd", fn = user_show_solution_details),
                        list_recipes = dict(value = "vr", fn = user_list_recipes),
                        list_recipes_dtls = dict(value = "vrd", fn = user_show_recipes_details),
                        add_cmp = dict(value = "ac", fn = user_add_cmpd),
                        add_sol_from_recipe = dict(value = "ar", fn = user_add_sol_from_recipe)
                        )

        looping_input(
            select_main_menu, 
            #"re[P]ository | [V]iew / list items | [R]ecipes | [q]uit ",
            """
------------------------------------------------------
                       MAIN MENU
------------------------------------------------------
Add
     ac: a compound
     ar: a solution/mixture based on recipe

Remove

Recipes
      r: binary mixtrue, optimize for 1 component

View
     vs: available solutions/mixtures
     vd: details of a solution/mixture
     vc: available components
     vr: available recipes
    vrd: details of a recipe

 exit,q: exit program
------------------------------------------------------
>>>
""",
            mmparams
            )



    def open(self, path):
        try:
            rep = Solution_Repository.fromJSON(file = path)
            with open(f'{path}.bak', 'w') as f:
                rep.toJSON(f, indent = 2)
            if rep is None:
                print(f'repository at:\n{path}\n results in NULL')
            return rep
        except Exception as e:
            print(f'while opening solution repository at:\n   `{path}`\nfollowing error occured:\n   {str(e)}')
            return None

    def add_solution_from_recipe(self):
        if len(self.recipes.keys()) == 0:
            print("no recipes created yet!!! Will stop now ... ")
            return

        choice = self.make_a_choice([x for x in self.recipes ])
        if choice is None:
            print("no recipe selected!!! Will stop now ... ")
            return
        
        a_recipe = self.recipes[choice]
        sol_1_name = a_recipe["sol_1"].name
        sol_2_name = a_recipe["sol_2"].name
        all_solutions = self.rep.list_solution_names()

        if not (sol_1_name in all_solutions and sol_2_name in all_solutions):
            print(f'{sol_1_name} or\n{sol_2_name}\n are not in repository anymore!!! Will stop now ...')
            

        def cancel_add_by_recipe(an_input, params):
            params["exit"] = True
            return True
        def is_valid_sol_name(an_input, params):
            sol_names = self.rep.list_solution_names()
            status = an_input not in sol_names
            if status:
                params["name"] = an_input
            else:
                print(f'solution with name: `{an_input}` is already defined, choose unique name!!!')
            return status
        
        mixt_params = dict(mixture_type = "binary_mix",
                            solute = sol_1_name,
                            solvent = sol_2_name,
                            v_ro = 1.0,
                            exit = False
                           )
         
        if not mixt_params["exit"]:

            select_name = dict(
                        main = is_valid_sol_name, 
                        exit = dict(value = "exit", fn = cancel_add_by_recipe), 
                        cancel = dict(value = "cancel", fn = cancel_add_by_recipe), 
                        cancel2 = dict(value = "c", fn = cancel_add_by_recipe)
                        )

            looping_input(
                select_name, 
                "Enter name for new solution>>>",
                mixt_params
                )

        def get_mass(sol_name, mass_key, params):
            if not params["exit"]:
                def is_ok_mass_value(an_input, params):
                    try:
                        val = float(an_input)
                        params[mass_key] = val
                        return True
                    except Exception:
                        print("Enter float number for mass value")
                        return False

                select_value = dict(
                            main = is_ok_mass_value,
                            exit = dict(value = "exit", fn = cancel_add_by_recipe), 
                            cancel = dict(value = "cancel", fn = cancel_add_by_recipe), 
                            cancel2 = dict(value = "c", fn = cancel_add_by_recipe)
                            )

                looping_input(
                    select_value, 
                    f"Enter mass (in g) of `{sol_name}` used>>>",
                    params
                    )

        get_mass(mixt_params["solute"], "m_solute", mixt_params)
        get_mass(mixt_params["solvent"], "m_solvent", mixt_params)
        print("\nNext two entries are for density estimate ... ")
        get_mass("water (density reference)", "m_ro_water", mixt_params)
        get_mass(mixt_params["name"], "m_ro", mixt_params)

        if not mixt_params["exit"]:
            self.rep.add_item(mixt_params)

        if not mixt_params["exit"] and mixt_params["name"] in self.rep.list_solution_names():
            print(f'solution `{mixt_params["name"]}` successfully created!!!')
        else:
            print("no soltion was added ...")

        #    def binary_mix_recipe_dict_str(mixture, solute, solvent, m_solute, m_solvent):
        #        return '{\n' + \
        #        f'    \"mixture_type\": \"binary_mix\",\n' + \
        #        f'    \"name\": \"{mixture.name}_{datetime.datetime.now().strftime("%Y%m%d")}\",\n' + \
        #        f'    \"m_solute\":   _{m_solute:0.4f}_,\n' + \
        #        f'    \"m_solvent\":  _{m_solvent:0.4f}_,\n' + \
        #        f'    \"solvent\":   \"{solvent.name}\",\n' + \
        #        f'    \"solute\":    \"{solute.name}\",\n' + \
        #        f'    \"v_ro\":       _1.0_,\n' + \
        #        f'    \"m_ro_water\": _1.0_,\n' + \
        #        f'    \"m_ro\":       _{mixture.ro:0.5f}_,\n' + \
        #        f'    \"date\":       \"{datetime.datetime.now().strftime("%Y-%m-%d")}\"\n' + \
        #        '}'

 
    def list_solutions(self):
        components = self.rep.list_solution_names()
        self.print_lot_of_choices(self.compact_string(components))

    def list_components(self):
        components = self.rep.list_components()
        self.print_lot_of_choices(self.compact_string(components))

    def list_recipes(self):
        recipes = [ x for x in self.recipes ]
        self.print_lot_of_choices(self.compact_string(recipes))

    def compact_string(self, options, show_index = True, width = 4, terminal_width = 80):
        if options is None or len(options) == 0:
            return ["No choices available; (use `c` to cancel prompt, if needed)"]

        max_width = max(list(map(len, options))) 
        total_width = max_width + width + 2
        columns = max(1, terminal_width // total_width)
        rows = len(options) // columns + ( 0 if len(options) % columns == 0 else 1 )
        #print(f'rows: {rows}, cols: {columns}')

        def temp_pretty_entry(i):
            if columns == 1 and total_width > terminal_width:
                return self.format_entry(
                                    options[i][0:terminal_width - width - 2],
                                    terminal_width,
                                    show_index,
                                    i,
                                    width)
            else:
                return self.format_entry(
                                    options[i],
                                    max_width,
                                    show_index,
                                    i,
                                    width)
        def make_row(i):
            index = [ x for x in range(len(options)) if x % rows == i ] 
            return functools.reduce(
                                lambda x, y: x + y, 
                                list(map(temp_pretty_entry, index)),
                                '')
            
        row_strings = [make_row(i) for i in range(rows)]
        #print(row_strings)

        #return functools.reduce(lambda x,y: x + '\n' + y, row_strings)
        return row_strings

    def format_entry(self, entry, width, show_index = True, index = -1, index_width = 4, sep = ':'):
        return f'{index: {index_width}d}: {entry:<{width}}'

    def print_lot_of_choices(self, choices, n = 30):
        k = ''
        for i,ch in enumerate(choices):
            print(ch)
            if i % n == n - 1:
                k = input("press enter to continue")
                if k == "q":
                    break
                if len(k) > 1 and k[0] == 'f':
                    break
        return k

    def make_a_choice(self, choices, prompt = None):
        if prompt is not None:
            print(prompt)
        compact_choices = self.compact_string(choices)
        out = self.print_lot_of_choices(compact_choices)
        while len(out) > 1 and out[0] == 'f':
            reduced_compact_choices = filter(lambda x: out[1:] in x, compact_choices)
            out = self.print_lot_of_choices(reduced_compact_choices)
        if prompt is not None:
            print(prompt)
        k = input("enter an index of item you want to select >>>")
        try:
            index = int(k)
            if index >= 0 and index < len(choices):
                return choices[index]
            else:
                print("no valid selection; nothing selected")
                return None
        except Exception:
            return None
        
    def pick_a_solution(self, prompt: str):
        #print(prompt)
        return self.make_a_choice(self.rep.list_solution_names(), prompt)

    def show_solution_details(self, label):
        #components = self.rep.list_solution_names()
        #label = components[2]
        print(self.solution_details_pretty(self.rep.items[label]))

    def show_solution_details_w_selection(self):
        self.show_solution_details(self.make_a_choice(self.rep.list_solution_names()))

    def solution_details_pretty(self, solution):
        def mk_str(v1, v2):
            return f'{v1:>40}   {v2}'
        headers = ["component name", "mass fraction (w)"]
        main = f'name: `{solution.name}`'
        details_ro = [mk_str("ro, g/ml", solution.ro)]
        details = [mk_str(solution.components[x].label, solution.components[x].w) for x in solution.components.keys()]
        details_hdr = [mk_str(headers[0], headers[1])]
        return functools.reduce(lambda x,y: x + '\n' + y, details_ro + details_hdr + details, main)

    def show_recipe_details(self, label):
        #print(self.recipes.keys())
        if label in self.recipes.keys():
            print(self.recipe_details_pretty(self.recipes[label]))

    def show_recipe_details_w_selection(self):
        self.show_recipe_details(self.make_a_choice([x for x in self.recipes]))

    def recipe_details_pretty(self, recipe):
        return recipe["recipe"]

    def add_new_compound(self):
        nc_params = dict(exit = False)

        def cancel_add_new_cmpd(an_input, params):
            params["exit"] = True
            return True
        def is_valid_new_cmpd_name(an_input, params):
            existing_components = self.rep.list_components()
            status = an_input not in existing_components and len(an_input) >= 3
            if status:
                params["name"] = an_input
            else:
                if an_input in existing_components:
                    print(f'There is already component with name `{an_input}`. Specify different name!!!')
                else:
                    print(f'Name `{an_input}` is too short, specify at least 3 symbols!!!')
            return status

        if not nc_params["exit"]:

            select_name = dict(
                        main = is_valid_new_cmpd_name, 
                        exit = dict(value = "exit", fn = cancel_add_new_cmpd), 
                        cancel = dict(value = "cancel", fn = cancel_add_new_cmpd), 
                        cancel2 = dict(value = "c", fn = cancel_add_new_cmpd)
                        )

            looping_input(
                select_name, 
                "Enter name for new compound>>>",
                nc_params
                )

        def is_valid_ro_value(an_input, params):
            try:
                val = float(an_input)
                status = val < 3 and val > 0.5
                if status:
                    params["ro"] = val
                else:
                    print(f'density value `{val}` is out of range 0.5..3.0')
                return status
            except Exception:
                print("Enter a float value!!!")
                return False
        
        if not nc_params["exit"]:

            select_ro = dict(
                        main = is_valid_ro_value, 
                        exit = dict(value = "exit", fn = cancel_add_new_cmpd), 
                        cancel = dict(value = "cancel", fn = cancel_add_new_cmpd), 
                        cancel2 = dict(value = "c", fn = cancel_add_new_cmpd)
                        )

            looping_input(
                select_ro, 
                "Enter density (in g/ml) for new compound (use 1.0 if not known)>>>",
                nc_params
                )

        if not nc_params["exit"]:
            self.rep.add_item(dict(
                                mixture_type = "pure_compound",
                                label = nc_params["name"],
                                name = nc_params["name"],
                                ro = nc_params["ro"] 
                                ))
            print(f'new compound was created!\n{json.dumps(self.rep.items[nc_params["name"]].toDict(), indent = 2)}')
        else:
            print("Add new compound canceled! Nothing was added!")

    def create_recipe_w_dialog(self):
            
        fn_params = dict(exit = False)

        def exit_routine(an_input, params):
            params["exit"] = True
            
        def is_valid_selection_sol_1(an_input, params):
            choice = self.make_a_choice(self.rep.list_solution_names())
            if choice is None:
                return False
            else: 
                params["solution_1_label"] = choice
                return True

        select_sol_1 = dict(
                        main = is_valid_selection_sol_1, 
                        exit = dict(value = "exit", fn = exit_routine), 
                        cancel = dict(value = "cancel", fn = exit_routine), 
                        cancel2 = dict(value = "c", fn = exit_routine)
                        )

        looping_input(
            select_sol_1, 
            "\n # 1st solution/compound to be used for recipe optimization #\n ... press enter ...",
            fn_params
            )

        if not fn_params["exit"]: 

            def is_valid_selection_sol_2(an_input, params):
                choice = self.make_a_choice(self.rep.list_solution_names())
                if choice is None:
                    print("no proper selection was made!!!")
                    return False
                elif choice == params["solution_1_label"]:
                    print("both selected solutions are identical, choose different solution!!!")
                    return False
                else: 
                    params["solution_2_label"] = choice
                    return True

            select_sol_2 = dict(
                            main = is_valid_selection_sol_2, 
                            exit = dict(value = "exit", fn = exit_routine), 
                            cancel = dict(value = "cancel", fn = exit_routine), 
                            cancel2 = dict(value = "c", fn = exit_routine)
                            )
            looping_input(
                select_sol_2, 
                "\n # 2nd solution/compound to be used for recipe optimization #\n ... press enter ...",
                fn_params
                )

        if not fn_params["exit"]: 
            def is_valid_component_selection(an_input, params):
                components_to_use = self.rep.list_components(names = [
                                                                    fn_params["solution_1_label"],
                                                                    fn_params["solution_2_label"]
                                                                    ])
                choice = self.make_a_choice(components_to_use)
                if choice is None:
                    print("no proper component choice was made!!!")
                    return False
                else:
                    params["component"] = choice
                    return True

            select_comp = dict(
                            main = is_valid_component_selection, 
                            exit = dict(value = "exit", fn = exit_routine), 
                            cancel = dict(value = "cancel", fn = exit_routine), 
                            cancel2 = dict(value = "c", fn = exit_routine)
                            )
            looping_input(
                select_comp, 
                "# component to be used for concentration optimization #\n ... press enter ...",
                fn_params
                )

        if not fn_params["exit"]: 
            def is_valid_conc_type(an_input, params):
                choice = self.make_a_choice(["m/m", "m/v"])
                if choice is None:
                    print("no proper concentration type was chosen!!!")
                    return False
                else:
                    params["concentration_type"] = choice
                    return True

            select_conc_type = dict(
                            main = is_valid_conc_type, 
                            exit = dict(value = "exit", fn = exit_routine), 
                            cancel = dict(value = "cancel", fn = exit_routine), 
                            cancel2 = dict(value = "c", fn = exit_routine)
                            )
            looping_input(
                select_conc_type, 
                "# concentration type to be used #\n ... press enter ...",
                fn_params
                )

        if not fn_params["exit"]: 
            def is_valid_target_value(an_input, params):
                try:
                    value = float(an_input)
                    params["target"] = value
                    return True
                except Exception:
                    print("please enter float value!")
                    return False

            select_trg_val = dict(
                            main = is_valid_target_value,
                            exit = dict(value = "exit", fn = exit_routine), 
                            cancel = dict(value = "cancel", fn = exit_routine), 
                            cancel2 = dict(value = "c", fn = exit_routine)
                            )

            range_string = "0 - 1" if fn_params['concentration_type'] == "m/m" else "0 - ro in g/ml"
            looping_input(
                select_trg_val, 
                f"Enter target concentration\n   available range: ({range_string}) for `{fn_params['concentration_type']}`\n>>>",
                fn_params
                )


        if not fn_params["exit"]: 
            def is_valid_amount(an_input, params):
                try:
                    value = float(an_input)
                    params["target_amount"] = value
                    return True
                except Exception:
                    print("please enter float value!")
                    return False

            select_trg_val = dict(
                            main = is_valid_amount,
                            exit = dict(value = "exit", fn = exit_routine), 
                            cancel = dict(value = "cancel", fn = exit_routine), 
                            cancel2 = dict(value = "c", fn = exit_routine)
                            )

            unit_string = "g" if fn_params['concentration_type'] == "m/m" else "ml"
            looping_input(
                select_trg_val, 
                f"Enter target amount in {unit_string} >>>",
                fn_params
                )

        print(fn_params)
        sol, recipe = create_recepie_for(
                            self.rep.items[fn_params["solution_1_label"]],
                            self.rep.items[fn_params["solution_2_label"]],
                            fn_params["component"],
                            fn_params["target"], 
                            fn_params["concentration_type"],
                            fn_params["target_amount"]
                            )
        print(recipe)

        
        
        recipe_dict = dict(
            sol_1 = self.rep.items[fn_params["solution_1_label"]],
            sol_2 = self.rep.items[fn_params["solution_2_label"]],
            component = fn_params["component"],
            target = fn_params["target"], 
            concentration_type = fn_params["concentration_type"],
            target_amount = fn_params["target_amount"],
            solution = sol,
            recipe = recipe
                            )

        if sol is not None:
            def is_valid_recipe_label(an_input, params):
                status = len(an_input) >= 5 and an_input not in self.recipes.keys()
                if status:
                    self.recipes[an_input] = recipe_dict 
                return status

            select_label = dict(
                            main = is_valid_recipe_label,
                            exit = dict(value = "exit", fn = exit_routine), 
                            cancel = dict(value = "cancel", fn = exit_routine), 
                            cancel2 = dict(value = "c", fn = exit_routine)
                            )
            
            looping_input(
                select_label, 
                f"Enter label for this recipe (5 or more symbols) >>>",
                fn_params
                )


        
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
    is_m_v = concentration_type == "m/v"
    if is_m_v:
        #print(msg_str(concentration_type))
        res = minimize(optimize_by_m_v, np.array([0.5]), method = method, bounds = [(0, 1)])
    else: # assume target is mass fraction
        #print(msg_str("default"))
        res = minimize(optimize_by_weight_fraction, np.array([0.3]), method = method, bounds = [(0, 1)])

    if res is not None and res.success:
        x = res.x[0]
        resulting_mixture = temp_sol(x) 
        
        m_solute = x * amount_needed * (resulting_mixture.ro if is_m_v else 1)
        m_solvent = (1-x) * amount_needed * (resulting_mixture.ro if is_m_v else 1)

        indent_str = '    '  
        a_unit = 'ml' if is_m_v else 'g'
        params_use = params['m/v' if is_m_v else 'default']
        human_recipe_lst = [
                f'optimizing for',
                f'{indent_str}{indent_str}component: `{component}`',
                f'{indent_str}{indent_str}concentration type: `{params_use["concentration"]}`',
                f'{indent_str}{indent_str}target: {target} {params_use["units"]}',
                f'Quantity: {amount_needed:0.4f} {a_unit}',
                f'',
                f'Use:'
                f'{indent_str}{m_solute:0.4f} g of {sol1.name} and',
                f'{indent_str}{m_solvent:0.4f} g of {sol2.name}',
                f'',
                f'{indent_str}m({sol2.name})/m({sol1.name}): {(1 - x) / x: 0.6f}  g/g'
                ]
        
        human_recipe = functools.reduce(lambda x,y: x + '\n' + y, human_recipe_lst)
        #print("\n------- dictionary for inclusion in solution repository -------\n")
        #print(binary_mix_recipe_dict_str(resulting_mixture, sol1, sol2, m_solute, m_solvent)) 
        #print(f"\n------- target solution representation -------\n\n{json.dumps(resulting_mixture.toDict(), indent = 2)}")
        return temp_sol(x), human_recipe
    else:
        print('failed to create a recepie')
        print(res)
        return None, 'failed to create a recepie'

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


