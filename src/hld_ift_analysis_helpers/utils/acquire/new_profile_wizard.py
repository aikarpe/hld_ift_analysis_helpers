
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

