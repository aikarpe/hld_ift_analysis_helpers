
step_big = 1.0
step_small = 0.1


# Source - https://stackoverflow.com/a/77896609
# Posted by BitsAreNumbersToo, modified by community. See post 'Timeline' for change history
# Retrieved 2026-04-15, License - CC BY-SA 4.0

import tkinter as tk
from PIL import Image, ImageTk
import cv2 as cv

class Needle_Positioner_Selector:

    
    def __init__(self, next_image_fn, move_to_offset_fn, offset_init, window_title, window_size, sign = dict(x = -1, z = 1), steps = dict(small = 0.25, large = 1.0)):
        self.next_image = next_image_fn
        self.movement_fn = move_to_offset_fn
        print(self.movement_fn)
        self.offset_init = offset_init
        self.offset = self.offset_init.copy()
        self.resize = 4
        self.counter = 0
        self.directions = dict(
                            up    = dict(x = 0, z = 1),
                            down  = dict(x = 0, z = -1),
                            left  = dict(x = -1, z = 0),
                            right = dict(x = 1, z = 0),
                            REST  = dict(x = 0, z = 0)
                            )
        self.sign = sign
        self.steps = steps

        self.coords =  dict(
                            n_pnts = 0,
                            x = [0,0],
                            y = [0,0],
                            start = [0,0],
                            end = [0,0]
                            )
        self.wroot = self.create_gui(window_title, window_size)
        print("something something Needle_Positioner_Selector")
                            
    def new_offset(self, direction_str: str, step_type_str: str):
        delta = self.directions[direction_str]
        self.offset["x"] = self.offset["x"] + self.steps[step_type_str] * self.sign["x"] * delta["x"]
        self.offset["z"] = self.offset["z"] + self.steps[step_type_str] * self.sign["z"] * delta["z"]

        return self.offset
        


    def create_gui(self, window_title, window_size):
        # Create Tkinter window
        root = tk.Tk()
        self.wroot = root
        print(f'after def: root: {self.wroot}')
        root.title(window_title)
        root.geometry(window_size)
        msg_value = """
           a: accept current selection, move on with scan
           c: cancel and used initial offset, move on with scan
        
               lowercase: step size: 0.25
               upppercase: step size: 1.0
        
           i: needle up
           k: needle down
           j: needle left
           l: needle right
         ==============================================
                                           [i]
           [a]                          [j][k][l]
                    [c]                            
         ==============================================
        
          *** move needle to optimal poisiton for acquisition ***
        
         click on image to select ROI for autofocusing!!!
        """
           
        msg = tk.Message(root, text = msg_value)
        msg.pack()
    
        num_clicks = tk.IntVar() #0)
        num_clicks.set(0)
        def five_clicks(*args):
            #global num_clicks
            if not num_clicks.get() % 5:
                print(f'after %5 clicks: {self.coords}')
    
        num_clicks.trace_add('write', five_clicks)
        # Add image
        image_label = self.add_image_to_gui(root)
        root.after(1, self.update_image, root, image_label)
    
        root.bind("a", self.close)
        root.bind("c", self.close_and_use_orginal_offset)
        
        root.bind("j", lambda x: self.movement_fn(self.new_offset("left", "small")))   #dict(string = "left")))
        root.bind("J", lambda x: self.movement_fn(self.new_offset("left", "large")))   #dict(string = "left")))
        root.bind("k", lambda x: self.movement_fn(self.new_offset("down", "small")))   #dict(string = "down")))
        root.bind("K", lambda x: self.movement_fn(self.new_offset("down", "large")))   #dict(string = "down")))
        root.bind("l", lambda x: self.movement_fn(self.new_offset("right", "small")))   #dict(string = "right")))
        root.bind("L", lambda x: self.movement_fn(self.new_offset("right", "large")))   #dict(string = "right")))
        root.bind("i", lambda x: self.movement_fn(self.new_offset("up", "small")))   #dict(string = "up")))
        root.bind("I", lambda x: self.movement_fn(self.new_offset("up", "large")))   #dict(string = "up")))
    
        #w.bind('<Button-1>', getcoord)
        image_label.bind('<Button-1>', lambda x: self.getcoord(x, num_clicks))
    
        # Start Tkinter event loop
        root.mainloop()
    
        print(f'root: {root}')
        return root, num_clicks

    def add_image_to_gui(self, root):
        image_label = tk.Label(root)
        image_label.pack()
        return image_label

    def update_image(self, root, image_label):
        image_raw = self.next_image() #cv.imread(function_that_yields_new_image_paths(), cv.IMREAD_COLOR)
        im_shp = image_raw.shape
        im_temp1 = cv.resize(image_raw, (im_shp[1] // self.resize, im_shp[0] // self.resize)) 
        im_temp2 = cv.rectangle(im_temp1, self.coords["start"], self.coords["end"], (0,0,255), 1) if self.coords["n_pnts"] == 2 else im_temp1
        tk_image = ImageTk.PhotoImage(image = Image.fromarray(im_temp2))
        image_label.configure(image=tk_image)
        image_label.image = tk_image
        root.after(300, self.update_image, root, image_label)

    #> def function_that_yields_new_image_paths():
    #>     #import numpy
    #>     global counter
    #>     a_path = files[counter % len(files)]
    #>     counter = counter + 1
    #>     return a_path

    def move_direction(self, offset):
        print(f'offset: {offset}')
    
    def getcoord(self, event, num_clicks):
        #global Cx, Cy #, num_clicks
        self.coords
        if self.coords["n_pnts"] < 2:
            index_use = self.coords["n_pnts"]
            self.coords["x"][index_use] = event.x 
            self.coords["y"][index_use] = event.y 
            print(f'x = {self.coords["x"][index_use]}; y = {self.coords["y"][index_use]}')
            self.coords["n_pnts"] = index_use + 1
            if index_use == 1:
                self.coords["start"] = [min(self.coords["x"]), min(self.coords["y"])]
                self.coords["end"] = [max(self.coords["x"]), max(self.coords["y"])]
        else:
            self.coords["n_pnts"] = 0 # remove last 2 points
    
    
        num_clicks.set(num_clicks.get() + 1)
        

    def close(self, *args):
        print(f'root c: {self.wroot}')
        self.wroot.destroy()
    def close_and_use_orginal_offset(self, *args):
        print(f'root cando: {self.wroot}')
        self.offset = self.offset_init
        self.wroot.destroy()
    
    def roi(self):
        return dict(
                min_x = self.resize * self.coords["start"][0],
                max_x = self.resize * self.coords["end"][0],
                min_y = self.resize * self.coords["start"][1],
                max_y = self.resize * self.coords["end"][1]
                )

if __name__ == "__main__":
    files = [
            "C:/Users/agaiosa/code/temp/00025.jpg",
            "C:/Users/agaiosa/code/temp/00090.jpg",
            "C:/Users/agaiosa/code/temp/00150.jpg"
            ]

    counter = 0
    def next_im():
        global counter
        path = files[counter % len(files)]
        image_raw = cv.imread(path, cv.IMREAD_COLOR)
        counter += 1
        return image_raw

    #Cx, Cy = 0, 0

    nps = Needle_Positioner_Selector(next_im, lambda x: print(x), dict(x = 0, y = 0, z = 0),    window_title = "Simple GUI", window_size = "800x800" ) #"1495x1020"

    print(nps.offset)
    print(nps.roi())

    #root, num_clicks = create_gui(window_title, window_size)
 

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
