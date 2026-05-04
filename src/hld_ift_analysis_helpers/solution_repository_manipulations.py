# object to modify solution repository with user input

# open/save/close sol rep
# list components/solutions
# generate recipe
# add substance/binary mixture

main_menu = ["Open/Save/Close repository", 
             "list items",
             "generate recipes",
             "add/remove items"
             ]

modify_rep_menu = ["Open", "Save", "Close"]
list_menu = ["components", "mixtures/solutions", "inspect_item"]
add_remove_menu = ["Add", "Remove"]

def open_user():
    if sol_rep is not None:
        close_user dlg
        if canceled: return 
        
        open_user dlg
        if canceled: return
        else:
            open selected

def close_user():
        # prompt saving
        ask_save/discard/cancel
        if wants to save
            save_user
            close
        elif wants to cancel operation
            return
        else
            close

def save_user():
        if sol_rep is not None
            save sol_rep

def list_components():
    k = input("filter>")
    a_list = get list
    apply k to a_list
    create menu with numbers

def list_solutions():
        get list
        print with numbers

def inspect_item():
        list_solutions()
        get selection
        pick selection
        show selection

def 
