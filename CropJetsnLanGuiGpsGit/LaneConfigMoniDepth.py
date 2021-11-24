#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import threading
if sys.version_info[0] == 2:  # Just checking your Python version to import Tkinter properly.
    from Tkinter import *
else:
    from tkinter import *
    from tkinter import ttk

from tools.CtrlGuiDepth import GUIDepth
from PIL import ImageTk, Image

from argparse import ArgumentParser as ArgParse

#https://redhulimachinelearning.com/python/pack-place-and-grid-in-tkinter/
"""
https://stackoverflow.com/questions/7966119/display-fullscreen-mode-on-tkinter
fullscreen window. Pressing Escape resizes the window to '200x200+0+0' by default. 
move or resize the window, Escape toggles between the current geometry and the previous geometry.
"""

class FullApp(object):

    def __init__(self, master, **kwargs):
        self.root = master#Tk()
        self.root.attributes('-zoomed', True)  # This just maximizes it so we can see the window. It's nothing to do with fullscreen.
        self.frame = Frame(self.root)
        self.frame.pack()
        self.state = False
        self.root.bind("<Escape>", self.toggle_fullscreen)
        self.root.bind("<F11>", self.end_fullscreen)
        self.root.config(bg="skyblue")
        self.gg = GUIDepth(self.root)

    def toggle_fullscreen(self, event=None):
        self.state = not self.state  # Just toggling the boolean
        self.root.attributes("-fullscreen", self.state)
        return "break"

    def end_fullscreen(self, event=None):
        self.state = False
        self.root.attributes("-fullscreen", False)
        return "break"
        
    def monicols(self,AllArgs):       

        def startbuttoncallback():
            self.gg.set_status("Starting...")
            self.gg.startproces(AllArgs)
        self.gg.add_button("Start !", startbuttoncallback)  

        def pausebuttoncallback():
            self.gg.set_status("Paused !")
        self.gg.add_button("Pause !", pausebuttoncallback)
        def listboxcallback(text):
            self.g.set_status("Method Select: '{0}'".format(text))
        self.gg.add_listbox(["Method 1", "Method 2", "Method 3"], listboxcallback)
        # g.add_listbox(["A", "B", "C"], listboxcallback)
        def scalecallback(text):
            self.gg.set_status("scale value: '{0:.2}'".format(text))
        self.gg.add_scale("Scale me!", scalecallback)
        self.gg.add_label("Text display.\nThat was a line break.\nThat was another.")
        bar_callback = self.gg.add_bar()
        # We need a mutable object to access it from the callback, so we use a list
        bar_value = [0.0]
        def increase_bar():
            bar_value[0] += 0.1
            if bar_value[0] > 1.0:
                bar_value[0] = 0.0
            bar_callback(bar_value[0])
        self.gg.add_button("Increase threshold..", increase_bar)
        entry = self.gg.add_labelentry("Type:", content = "Data  is ready.")
        def get_entry():
            self.gg.set_status(entry.get())
        self.gg.add_button(" Info to entry", get_entry)
        rating_scale = self.gg.add_rating_scale("Image Chosen:", 2, "RGB", "Depth")
        def get_rating():
            self.gg.set_status(rating_scale.get())
        self.gg.add_button("Select Image Type", get_rating)
        self.gg.add_button("Quit", lambda: self.gg.destroy())
        def run_when_started():
            self.gg.set_status("Ready to be executed")
        # g.center()
        self.gg.run(run_when_started)


def main(args):

    root = Tk() # create root window
    root.title("Lane Configuring Console")
    app = FullApp(root)
    # app.tk.mainloop()
    app.monicols(args)

def get_parser():
    
    parser =ArgParse()
    parser.add_argument('--numL', type=int, help="Initial Setting of total number of Lane to track !",default=4) # simplifed : 6 , RGB and depth: 4
    parser.add_argument('--pipL', type=int, help="Detecting Pipe Line Choosing: simplified 1, complex- RGB (2) and Depth (3) !",default=3)

    return parser

if __name__ == "__main__":

    args = get_parser().parse_args()
    main(args)
