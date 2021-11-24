import sys
import threading
if sys.version_info[0] == 2:  # Just checking your Python version to import Tkinter properly.
    from Tkinter import *
else:
    from tkinter import *
    from tkinter import ttk

from tools.CtrlGui import GUI
from PIL import ImageTk, Image

#https://redhulimachinelearning.com/python/pack-place-and-grid-in-tkinter/
"""
https://stackoverflow.com/questions/7966119/display-fullscreen-mode-on-tkinter
fullscreen window. Pressing Escape resizes the window to '200x200+0+0' by default. 
move or resize the window, Escape toggles between the current geometry and the previous geometry.
"""
class FullScreenApp(object):
    def __init__(self, master, **kwargs):
        self.root=master
        pad=3
        self._geom='200x200+0+0'
        master.geometry("{0}x{1}+0+0".format(
            master.winfo_screenwidth()-pad, master.winfo_screenheight()-pad))
        master.bind('<Escape>',self.toggle_geom)            
    def toggle_geom(self,event):
        geom=self.root.winfo_geometry()
        print(geom,self._geom)
        self.root.geometry(self._geom)
        self._geom=geom

    def run(self):

        # root = Tk() # create root window
        # root.title("Lane Configuring Console")
        # root.maxsize(1280, 960) 
        # root.config(bg="skyblue")

        # root.attributes("-fullscreen", True)  # substitute `Tk` for whatever your `Tk()` object is called
        # root.overrideredirect(True)
        # root.geometry("{0}x{1}+0+0".format(root.winfo_screenwidth(), root.winfo_screenheight()))
        

        # Create left and right frames
        self.left_frame = Frame(self.root, width=960, height=640, bg='grey')
        self.left_frame.pack(side='left', fill='both', padx=8, pady=4, expand=True)

        self.right_frame = Frame(self.root, width=256, height=640, bg='grey')
        self.right_frame.pack(side='right', fill='both', padx=8, pady=4, expand=True)

        # right_frame = Frame(root, width=650, height=400, bg='grey')
        # right_frame.pack(side='right', fill='both', padx=10, pady=5, expand=True)
        self.image=Image.open("/home/chfox/ARWAC/Essa-0/20190516_142759.jpg")
        self.width, self.height = self.image.size
        # Create frames and labels in left_frame
        # Label(left_frame, text="Output Stream").pack(side='top', padx=4, pady=4)
        self.height_l=int(self.height*0.5)
        self.width_l = int(self.width*0.5)
        self.image_l = self.image.resize((self.height_l, self.width_l), Image.ANTIALIAS) ## The (250, 250) is (height, width)
        self.image_l = ImageTk.PhotoImage(self.image_l)
        # Label(left_frame, image=image_l).pack(fill='both', padx=4, pady=4)

        self.canvas = Canvas(self.left_frame, width = int(self.width*0.5), height = int(self.height*0.5))  
        self.canvas.pack()  
        self.canvas.create_image(20, 20, anchor=NW, image=self.image_l) 
        #####################################################
        # Label(right_frame, text="Input Stream").pack(side='top', padx=5, pady=5)       
        #large_image = original_image.subsample(2,2)
        self.height_r=int(self.height*0.05)
        self.width_r = int(self.width*0.05)
        self.image_r = self.image.resize((self.height_r, self.width_r), Image.ANTIALIAS) ## The (250, 250) is (height, width)
        self.image_r = ImageTk.PhotoImage(self.image_r)

        # Label(right_frame, image=image_r).pack(fill='both', padx=5, pady=5)
        self.canvas = Canvas(self.right_frame, width = int(self.width*0.05), height = int(self.height*0.05))  
        self.canvas.pack()  
        self.canvas.create_image(20, 20, anchor=NW, image=self.image_r)



        


        self.tool_bar = Frame(self.right_frame, width=90, height=185, bg='lightgrey')
        self.tool_bar.pack(side='right', fill='both', padx=5, pady=5, expand=True)

        self.filter_bar = Frame(self.right_frame, width=90, height=185, bg='lightgrey')
        self.filter_bar.pack(side='left', fill='both', padx=5, pady=5, expand=True)

        def clicked():
            '''if button is clicked, display message'''
            print("Clicked.")
        # Example labels that serve as placeholders for other widgets 
        Label(self.tool_bar, text="Tools", relief=RAISED).pack(anchor='n', padx=5, pady=3, ipadx=10)
        Label(self.filter_bar, text="Filters", relief=RAISED).pack(anchor='n', padx=5, pady=3, ipadx=10)

        # For now, when the buttons are clicked, they only call the clicked() method. We will add functionality later.
        Button(self.tool_bar, text="Select", command=clicked).pack(padx=5, pady=5)
        Button(self.tool_bar, text="Crop", command=clicked).pack(padx=5, pady=5)
        Button(self.tool_bar, text="Rotate & Flip", command=clicked).pack(padx=5, pady=5)
        Button(self.tool_bar, text="Resize", command=clicked).pack(padx=5, pady=5)
        Button(self.filter_bar, text="Black & White", command=clicked).pack(padx=5, pady=5)

        # app=FullScreenApp(root)
        self.root.mainloop()

class Fullscreen_Window(object):

    def __init__(self, master, **kwargs):
        self.root = master#Tk()
        self.root.attributes('-zoomed', True)  # This just maximizes it so we can see the window. It's nothing to do with fullscreen.
        self.frame = Frame(self.root)
        self.frame.pack()
        self.state = False
        self.root.bind("<Escape>", self.toggle_fullscreen)
        self.root.bind("<F11>", self.end_fullscreen)
        self.root.config(bg="skyblue")

    def toggle_fullscreen(self, event=None):
        self.state = not self.state  # Just toggling the boolean
        self.root.attributes("-fullscreen", self.state)
        return "break"

    def end_fullscreen(self, event=None):
        self.state = False
        self.root.attributes("-fullscreen", False)
        return "break"
        
    def run(self):

        # root = Tk() # create root window
        # root.title("Lane Configuring Console")
        # root.maxsize(1280, 960) 
        # root.config(bg="skyblue")

        # root.attributes("-fullscreen", True)  # substitute `Tk` for whatever your `Tk()` object is called
        # root.overrideredirect(True)
        # root.geometry("{0}x{1}+0+0".format(root.winfo_screenwidth(), root.winfo_screenheight()))
        

        # Create left and right frames
        self.left_frame = Frame(self.root, width=960, height=640, bg='grey')
        self.left_frame.pack(side='left', fill='both', padx=8, pady=4, expand=True)

        self.right_frame = Frame(self.root, width=256, height=640, bg='grey')
        self.right_frame.pack(side='right', fill='both', padx=8, pady=4, expand=True)

        # right_frame = Frame(root, width=650, height=400, bg='grey')
        # right_frame.pack(side='right', fill='both', padx=10, pady=5, expand=True)
        self.image=Image.open("/home/dom/ARWAC/WeedRobLanCtrl/InitImages/20190516_142759.jpg")
        width, height = self.image.size
        # Create frames and labels in left_frame
        Label(self.left_frame, text="Output Stream").pack(side='top', padx=4, pady=4)
        self.image_l = self.image.resize((int(height*0.5), int(width*0.5)), Image.ANTIALIAS) ## The (250, 250) is (height, width)

        self.image_l = ImageTk.PhotoImage(self.image_l)
        Label(self.left_frame, image=self.image_l).pack(fill='both', padx=4, pady=4)

        # canvas = Canvas(left_frame, width = width, height = height)  
        # canvas.pack()  
        # canvas.create_image(20, 20, anchor=NW, image=image_l) 

        Label(self.right_frame, text="Input Stream").pack(side='top', padx=5, pady=5)
        # Label(left_frame, image=image_l).pack(fill='both', padx=5, pady=5)
        #large_image = original_image.subsample(2,2)
        self.image_r = self.image.resize((int(height*0.05), int(width*0.05)), Image.ANTIALIAS) ## The (250, 250) is (height, width)
        self.image_r = ImageTk.PhotoImage(self.image_r)
        Label(self.right_frame, image=self.image_r).pack(fill='both', padx=5, pady=5)


        tool_bar = Frame(self.right_frame, width=90, height=185, bg='lightgrey')
        tool_bar.pack(side='right', fill='both', padx=5, pady=5, expand=True)

        filter_bar = Frame(self.right_frame, width=90, height=185, bg='lightgrey')
        filter_bar.pack(side='left', fill='both', padx=5, pady=5, expand=True)

        def clicked():
            '''if button is clicked, display message'''
            print("Clicked.")
        # Example labels that serve as placeholders for other widgets 
        Label(tool_bar, text="Tools", relief=RAISED).pack(anchor='n', padx=5, pady=3, ipadx=10)
        Label(filter_bar, text="Filters", relief=RAISED).pack(anchor='n', padx=5, pady=3, ipadx=10)

        # For now, when the buttons are clicked, they only call the clicked() method. We will add functionality later.
        Button(tool_bar, text="Select", command=clicked).pack(padx=5, pady=5)
        Button(tool_bar, text="Crop", command=clicked).pack(padx=5, pady=5)
        Button(tool_bar, text="Rotate & Flip", command=clicked).pack(padx=5, pady=5)
        Button(tool_bar, text="Resize", command=clicked).pack(padx=5, pady=5)
        Button(filter_bar, text="Black & White", command=clicked).pack(padx=5, pady=5)

        # app=FullScreenApp(root)
        self.root.mainloop()


"""
def run():

    g = GUI()
    def buttoncallback():
        g.set_status("Button klicked!")
    g.add_button("Klick me!", buttoncallback)
    g.add_button("Klick me too!", buttoncallback)
    def listboxcallback(text):
        g.set_status("listbox select: '{0}'".format(text))
    g.add_listbox(["one", "two", "three"], listboxcallback)
    g.add_listbox(["A", "B", "C"], listboxcallback)
    def scalecallback(text):
        g.set_status("scale value: '{0:.2}'".format(text))
    g.add_scale("Scale me!", scalecallback)
    g.add_label("This is a label for text display.\nThat was a line break.\nThat was another.")
    bar_callback = g.add_bar()
    # We need a mutable object to access it from the callback, so we use a list
    bar_value = [0.0]
    def increase_bar():
        bar_value[0] += 0.1
        if bar_value[0] > 1.0:
            bar_value[0] = 0.0
        bar_callback(bar_value[0])
    g.add_button("Increase Bar", increase_bar)
    entry = g.add_labelentry("Type something:", content = "There's already something here.")
    def get_entry():
        g.set_status(entry.get())
    g.add_button("Set status to entry text", get_entry)
    rating_scale = g.add_rating_scale("Rate your experience:", 5, "Bad", "Good")
    def get_rating():
        g.set_status(rating_scale.get())
    g.add_button("Set status to rating", get_rating)
    g.add_button("Quit", lambda: g.destroy())
    def run_when_started():
        g.set_status("run_when_started() executed")
    g.center()
    g.run(run_when_started)

    """

'''Example of how to use the pack() method to create a GUI layout'''


# import tkinter and all its functions    

def run(root):

    # root = Tk() # create root window
    # root.title("Lane Configuring Console")
    # root.maxsize(1280, 960) 
    # root.config(bg="skyblue")

    # root.attributes("-fullscreen", True)  # substitute `Tk` for whatever your `Tk()` object is called
    # root.overrideredirect(True)
    # root.geometry("{0}x{1}+0+0".format(root.winfo_screenwidth(), root.winfo_screenheight()))
    

    # Create left and right frames
    left_frame = Frame(root, width=960, height=640, bg='grey')
    left_frame.pack(side='left', fill='both', padx=8, pady=4, expand=True)

    right_frame = Frame(root, width=256, height=640, bg='grey')
    right_frame.pack(side='right', fill='both', padx=8, pady=4, expand=True)

    # right_frame = Frame(root, width=650, height=400, bg='grey')
    # right_frame.pack(side='right', fill='both', padx=10, pady=5, expand=True)
    image=Image.open("/home/dom/ARWAC/WeedRobLanCtrl/InitImages/20190206_141310.jpg")
    image=Image.open("/home/dom/ARWAC/WeedRobLanCtrl/InitImages/20190516_142759.jpg")
    width, height = image.size
    # Create frames and labels in left_frame
    Label(left_frame, text="Output Stream").pack(side='top', padx=4, pady=4)
    image_l = image.resize((int(height*0.5), int(width*0.5)), Image.ANTIALIAS) ## The (250, 250) is (height, width)

    image_l = ImageTk.PhotoImage(image_l)
    Label(left_frame, image=image_l).pack(fill='both', padx=4, pady=4)

    # canvas = Canvas(left_frame, width = width, height = height)  
    # canvas.pack()  
    # canvas.create_image(20, 20, anchor=NW, image=image_l) 

    Label(right_frame, text="Input Stream").pack(side='top', padx=5, pady=5)
    # Label(left_frame, image=image_l).pack(fill='both', padx=5, pady=5)
    #large_image = original_image.subsample(2,2)
    image_r = image.resize((int(height*0.05), int(width*0.05)), Image.ANTIALIAS) ## The (250, 250) is (height, width)
    image_r = ImageTk.PhotoImage(image_r)
    Label(right_frame, image=image_r).pack(fill='both', padx=5, pady=5)


    tool_bar = Frame(right_frame, width=90, height=185, bg='lightgrey')
    tool_bar.pack(side='right', fill='both', padx=5, pady=5, expand=True)

    filter_bar = Frame(right_frame, width=90, height=185, bg='lightgrey')
    filter_bar.pack(side='left', fill='both', padx=5, pady=5, expand=True)

    def clicked():
        '''if button is clicked, display message'''
        print("Clicked.")
    # Example labels that serve as placeholders for other widgets 
    Label(tool_bar, text="Tools", relief=RAISED).pack(anchor='n', padx=5, pady=3, ipadx=10)
    Label(filter_bar, text="Filters", relief=RAISED).pack(anchor='n', padx=5, pady=3, ipadx=10)

    # For now, when the buttons are clicked, they only call the clicked() method. We will add functionality later.
    Button(tool_bar, text="Select", command=clicked).pack(padx=5, pady=5)
    Button(tool_bar, text="Crop", command=clicked).pack(padx=5, pady=5)
    Button(tool_bar, text="Rotate & Flip", command=clicked).pack(padx=5, pady=5)
    Button(tool_bar, text="Resize", command=clicked).pack(padx=5, pady=5)
    Button(filter_bar, text="Black & White", command=clicked).pack(padx=5, pady=5)

    # app=FullScreenApp(root)
    root.mainloop()

def main():

    root = Tk() # create root window
    root.title("Lane Configuring Console")
    # root.overrideredirect(True) # stuck on the screen
    # root.attributes('-zoomed', True)
    # root.maxsize(1280, 960) 
    # root.config(bg="skyblue")

    # root.wm_attributes("-fullscreen", True)  # substitute `Tk` for whatever your `Tk()` object is called
    app=FullScreenApp(root)
    # root.mainloop()
    app = Fullscreen_Window(root)
    # app.tk.mainloop()
    app.run()

if __name__ == '__main__':
    main()
