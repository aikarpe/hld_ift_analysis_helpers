counter = 0

step_big = 1.0
step_small = 0.1

files = [
        "C:/Users/agaiosa/code/temp/00025.jpg",
        "C:/Users/agaiosa/code/temp/00090.jpg",
        "C:/Users/agaiosa/code/temp/00150.jpg"
        ]


# Source - https://stackoverflow.com/a/77896609
# Posted by BitsAreNumbersToo, modified by community. See post 'Timeline' for change history
# Retrieved 2026-04-15, License - CC BY-SA 4.0

import tkinter as tk
from PIL import Image, ImageTk


resize = 4

def add_image_to_gui(root):
    image_label = tk.Label(root)
    image_label.pack()
    return image_label

def update_image(root, image_label):
    image_raw = Image.open(function_that_yields_new_image_paths())
    sz = image_raw.size
    image = image_raw.resize((sz[0] // resize, sz[1] // resize))
    tk_image = ImageTk.PhotoImage(image)
    image_label.configure(image=tk_image)
    image_label.image = tk_image
    root.after(1000, update_image, root, image_label)

def function_that_yields_new_image_paths():
    #import numpy
    global counter
    a_path = files[counter % len(files)]
    counter = counter + 1
    return a_path

def move_direction(offset):
    print(f'offset: {offset}')

def create_gui(window_title, window_size):
    # Create Tkinter window
    root = tk.Tk()
    root.title(window_title)
    root.geometry(window_size)
    msg_value = """
                   i
    a             j k l    
      c
    """
    msg = tk.Message(root, text = msg_value)
    msg.pack()

    num_clicks = tk.IntVar() #0)
    num_clicks.set(0)
    def five_clicks(*args):
        #global num_clicks
        if not num_clicks.get() % 5:
            print(f'after %5 clicks: {Cx=}, {Cy=}')

    num_clicks.trace_add('write', five_clicks)

    # Add image
    image_label = add_image_to_gui(root)
    root.after(1, update_image, root, image_label)

    root.bind("a", lambda x: move_direction(dict(string = "accept")))
    root.bind("c", lambda x: move_direction(dict(string = "cancel")))

    root.bind("j", lambda x: move_direction(dict(string = "left")))
    root.bind("k", lambda x: move_direction(dict(string = "down")))
    root.bind("l", lambda x: move_direction(dict(string = "right")))
    root.bind("i", lambda x: move_direction(dict(string = "up")))

    #w.bind('<Button-1>', getcoord)
    image_label.bind('<Button-1>', lambda x: getcoord(x, num_clicks))

    # Start Tkinter event loop
    root.mainloop()

    return root, num_clicks

def getcoord(event, num_clicks):
    global Cx, Cy #, num_clicks
    Cx, Cy = event.x, event.y
    print('X = ', Cx, '   Y=  ', Cy)
    num_clicks.set(num_clicks.get() + 1)




if __name__ == "__main__":
    Cx, Cy = 0, 0

    window_title = "Simple GUI"
    window_size = "800x600" #"1495x1020"

    root, num_clicks = create_gui(window_title, window_size)
 

#> def on_button_click(self, event=None): # command= takes a function with no arguments while .bind takes a function with one argument
#>     print("Clicked the button!")

#> root = tk.Tk()
#> button = tk.Button(root, text="Click me!", command=on_button_click)
#> root.bind("<Control-a>", on_button_click)


#==========================================

#> if __name__ == '__main__':
#> 
#>     Cx, Cy = 0, 0
#> 
#>     root = tk.Tk()
#>     num_clicks = tk.IntVar() #0)
#>     num_clicks.set(0)
#>     num_clicks.trace_add('write', five_clicks)
#> 
#>     w = tk.Canvas(root, width=500, height=500)
#>     w.pack()
#> 
#>     w.bind('<Button-1>', getcoord)
#> 
#>     root.mainloop()
