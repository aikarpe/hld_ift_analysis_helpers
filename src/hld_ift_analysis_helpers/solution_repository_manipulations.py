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

def select_item() 


# start program
#   open repository
#   make a copy of repository obj OR copy of its file (.bak)
#   
# main menu loop:
#   make a choice
#       if choice == exit
#           break loop
#       elif choice == add:
#           add menu
#       elif choice == delete:
#           delete_menu
#       elif choice == view:
#           view_menu
#       elif choice == recipe:
#           recipe_menu
#       elif choice == file_op:
#           file_op_menu
#
# end program
#   prompt: save changes y/n
#   if y:
#       save current state of sol rep
#   else:
#       restore initial from copy or bak

# add menu:
#   make a choice
#       if choice = add compound
#       elif choice = add mixture


# import item(s) from another repository
# save changes
# discard changes
# exit

# delete item from repository
# add new compound (1 component)
# add new mixture
# add new binary mixture:
# recipe: existing mixture
# recipe: from two mixtures

# menu_main
#   import, delete, add, recipe, view
