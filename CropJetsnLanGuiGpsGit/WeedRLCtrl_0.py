import sys
import threading

# from tkinter import ttk
from tools.CtrlGui_0 import GUI


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

def main():
    run()

if __name__ == '__main__':
    main()
