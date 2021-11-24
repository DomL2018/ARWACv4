import sys
import threading
import matplotlib as mpl
import os
# if os.environ.get('DISPLAY','') == '':
#     print('no display found. Using non-interactive Agg backend')
#     mpl.use('Agg')
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using :1')
    os.environ.__setitem__('DISPLAY', ':1')

if sys.version_info[0] == 2:  # Just checking your Python version to import Tkinter properly.
    from Tkinter import *
else:
    from tkinter import *
    from tkinter import ttk

from tools.CtrlGui import GUI
from PIL import ImageTk, Image


def run(root):

    g = GUI(root)

    def startbuttoncallback():
        g.set_status("Starting...")
    g.add_button("Start !", startbuttoncallback)  

    def pausebuttoncallback():
        g.set_status("Paused !")
    g.add_button("Pause !", pausebuttoncallback)
    def listboxcallback(text):
        g.set_status("Method Select: '{0}'".format(text))
    g.add_listbox(["Method 1", "Method 2", "Method 3"], listboxcallback)
    # g.add_listbox(["A", "B", "C"], listboxcallback)
    def scalecallback(text):
        g.set_status("scale value: '{0:.2}'".format(text))
    g.add_scale("Scale me!", scalecallback)
    g.add_label("Text display.\nThat was a line break.\nThat was another.")
    bar_callback = g.add_bar()
    # We need a mutable object to access it from the callback, so we use a list
    bar_value = [0.0]
    def increase_bar():
        bar_value[0] += 0.1
        if bar_value[0] > 1.0:
            bar_value[0] = 0.0
        bar_callback(bar_value[0])
    g.add_button("Increase threshold..", increase_bar)
    entry = g.add_labelentry("Type:", content = "Data  is ready.")
    def get_entry():
        g.set_status(entry.get())
    g.add_button(" Info to entry", get_entry)
    rating_scale = g.add_rating_scale("Image Chosen:", 2, "RGB", "Depth")
    def get_rating():
        g.set_status(rating_scale.get())
    g.add_button("Select Image Type", get_rating)
    g.add_button("Quit", lambda: g.destroy())
    def run_when_started():
        g.set_status("Ready to be executed")
    # g.center()
    g.run(run_when_started)

def main():
    root = Tk() # create root window
    root.title("Lane Configuring Console")
    # app = Fullscreen_Window(root)
    # app.tk.mainloop()
    # app.run()
    run(root)

if __name__ == '__main__':
    main()
