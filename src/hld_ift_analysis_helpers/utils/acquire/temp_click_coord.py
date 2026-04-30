import tkinter as tk


def getcoord(event):
    global Cx, Cy
    Cx, Cy = event.x, event.y
    print('X = ', Cx, '   Y=  ', Cy)
    num_clicks.set(num_clicks.get() + 1)


def five_clicks(*args):
    if not num_clicks.get() % 5:
        print(f'after %5 clicks: {Cx=}, {Cy=}')


if __name__ == '__main__':

    Cx, Cy = 0, 0

    root = tk.Tk()
    num_clicks = tk.IntVar() #0)
    num_clicks.set(0)
    num_clicks.trace_add('write', five_clicks)

    w = tk.Canvas(root, width=500, height=500)
    w.pack()

    w.bind('<Button-1>', getcoord)

    root.mainloop()
