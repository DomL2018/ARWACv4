import sys
import threading
import tkinter
import tkinter.ttk as ttk
class GUI:
    """Base class for a window to add widgets to.
       Attributes:
       GUI.root
           The root Tk object. Only use when you know what you are
           doing.
       GUI.width
           The width of scale, bar etc. widgets.
       GUI.style
           The ttk style when ttk is being used. Else None.
       GUI.frame
       GUI.statusbar
           Standard GUI widgets.
       GUI.threads
           A list of Thread objects created by GUI.
       GUI.exit
           Boolean flag whether to exit the application. To be
           set by callbacks and caught by GUI.check_exit().
    """
    def __init__(self, title = "Lanes Tracking GUI", width = 800):
        """Initialise. Call this first before calling any other tkinter routines.

           title is the window title to use.

           width is the width of scale, bar etc. widgets.
        """

        self.root = tkinter.Tk()

        # self.root = tkinter.Toplevel()

        self.root.title(title)

        self.width = width

        self.style = None

        self.style = ttk.Style()

        # TODO: mv frame _frame?

        self.frame = tkinter.Frame(master = self.root)

        self.frame.pack(fill = "both")

        # TODO: mv statusbar _statusbar?

        self.statusbar = tkinter.Label(master = self.frame,
                                       text = "Ctrl.gui (tkinter {0})".format(tkinter.TkVersion))

        self.statusbar.pack(padx = 10, pady = 5)

        self.threads = []

        self.exit = False

        return

    def set_status(self, text):
        """Display the new status text in the status bar. """
        self.statusbar["text"] = text
        return

    def center(self):
        """Center the root window on screen.
        """
        self.root.update()
        screenwidth = self.root.winfo_screenwidth()
        screenheight = self.root.winfo_screenheight()
        windowwidth = self.root.winfo_width()
        windowheight = self.root.winfo_height()
        self.root.geometry("+{0}+{1}".format(int(screenwidth / 2 - windowwidth / 2),
                                             int(screenheight / 2 - windowheight / 2)))

        return

    def check_exit(self):
        """Check whether to shut down the application. If not, waits and calls itself.

           This method is started by GUI.run() and runs in the
           main thread.
        """

        if self.exit == True:

            sys.stderr.write("check_exit() caught GUI.exit flag, calling destroy()" + "\n")

            self.destroy()

        else:
            # Check every 10 ms            #
            self.root.after(10, func = self.check_exit)
        return

    def run(self, function = None):
        """Start the exit checking function, and run the tkinter mainloop.

           function, if given, must be a callable which will be run
           in a thread once the application runs.
        """
        self.root.after(10, func = self.check_exit)
        if function is not None:
            self.root.after(10, func = self.get_threaded(function))
        self.root.mainloop()
        return

    def destroy(self):
        """Destroy this window in a safe manner.
        """
        if threading.current_thread() in self.threads:

            # Destroying from outside the main thread leads
            # to all sorts of problems. Set a flag instead
            # that will be caught by the main thread.
            #
            sys.stderr.write("destroy() called from callback thread, setting GUI.exit" + "\n")

            self.exit = True

            return

        # First, prevent accidental interaction
        #
        self.root.withdraw()

        sys.stderr.write("Attempting to join threads" + "\n")

        for thread in self.threads:

            sys.stderr.write("Joining thread '" + thread.name + "'" + "\n")

            thread.join()

        sys.stderr.write("All possible threads joined" + "\n")

        # [In Python using Tkinter, what is the difference between root.destroy() and root.quit()?](https://stackoverflow.com/questions/2307464/in-python-using-tkinter-what-is-the-difference-between-root-destroy-and-root)
        #
        self.root.destroy()
        return

    def get_threaded(self, callback):
        """Return a function that, when called, will run `callback` in a background thread.

           The thread will be registrered in GUI.threads upon start.
        """

        def threaded_callback():

            # Create a new Thread upon each call, so each Thread's
            # start() method is only called once.
            #
            callback_thread = threading.Thread(target = callback,
                                               name = "Thread-" + callback.__name__)

            self.threads.append(callback_thread)

            # This will return immediately.
            #
            callback_thread.start()

            return

        return threaded_callback

    def add_button(self, label, callback):
        """Add a button with label `label` calling `callback` with no arguments when clicked.

           The callback will run in a background thread.
        """

        tkinter.Button(master = self.frame,
                       text = label,
                       command = self.get_threaded(callback)).pack(padx = 10, pady = 5)

        return

    def add_listbox(self, stringlist, callback):
        """Add a listbox.

           Klicking an entry in listbox will call `callback(entry)`,
           where `entry` is a string.
        """

        # TODO: What if the first item is the one to be selected, and no click happens? Wouldn't it be better to return something to query the selected item from, as done in labelentry?

        stringvar = tkinter.StringVar(value = tuple(stringlist))

        listbox = tkinter.Listbox(master = self.frame,
                                  height = len(stringlist),
                                  listvariable = stringvar)

        listbox.pack(padx = 10, pady = 5)

        # TODO: Make callback threaded
        #
        def callbackwrapper(event):
            callback(event.widget.get(event.widget.curselection()[0]))

        listbox.bind("<<ListboxSelect>>", callbackwrapper)

        return

    def add_scale(self, scalelabel, callback):
        """Add a scale.

           Moving the scale will call `callback(value)`, where `value`
           is a float 0..1.
        """

        # if USING_TTK:

            # No label option for ttk Scale
            #
        self.add_label(scalelabel)

            # No showvalue option for ttk Scale
            #
        value_label = tkinter.Label(master = self.frame,text = "sliding..")

        value_label.pack(padx = 5, pady = 5)

        # TODO: Make callback threaded
        #
        def callbackwrapper(value):

            # ttk.Scale calls back with a float string
            #
            value_label["text"] = int(float(value) * 10)
            callback(float(value))

        self.style.configure("TScale",orient = "horizontal",sliderlength = 10)

        # tkinter.Scale(master = self.frame,length = self.width,command = callbackwrapper).pack(padx = 10, pady = 5)
        w = tkinter.Scale(master= self.frame, from_=10, to=-10,command = callbackwrapper)
        w.pack(padx = 10, pady = 5)        
        return

    def add_label(self, text):
        """Add a label.
        """
        tkinter.Label(master = self.frame,
                      text = text).pack(padx = 10, pady = 5)
        return

    def add_bar(self):
        """Add a bar for progress or meter display.

           Returns a function to be called with a float 0..1.
        """
        # if USING_TTK:

            # This is not a standard widget, so we get it from
            # ttk directly.
            #
        bar = ttk.Progressbar(master = self.frame,
                                          orient = "horizontal",
                                          length = self.width,
                                          mode = "determinate",
                                          maximum = 100,
                                          value = 0)

        bar.pack(padx = 10, pady = 5)
        def callback(value):
            bar["value"] = int(value * 100)
            return
        return callback

    def add_canvas(self, width, height):
        """Add a canvas and return it. """

        canvas = tkinter.Canvas(master = self.frame,
                                width = width,
                                height = height)
        canvas.pack(padx = 10, pady = 5)
        return canvas

    def add_labelentry(self, text, width = 16, content = ""):
        """Create a labelled entry, and return the entry.
           width is the number of characters to show.
           content is a string to show in the entry.
           Call entry.get() to get its current content.
        """ 
        labelentry = tkinter.Frame(master = self.frame)
        tkinter.Label(master = labelentry,
                          text = text).pack(side = tkinter.LEFT)

        entry = tkinter.Entry(master = labelentry,
                                  width = width)

        entry.insert(0, content)
        entry.pack()
        labelentry.pack()
        return entry

    def add_rating_scale(self, scalelabel, count, worst_label, best_label):
        """Add a rating scale, with `count` options to choose from.

           Returns an object whose get() method will yield the integer
           index of the selected item, starting with 0.
        """

        tkinter.Label(master = self.frame,
                      text = scalelabel).pack(padx = 10, pady = 5)

        rating_scale_frame = tkinter.Frame(master = self.frame)

        tkinter.Label(master = rating_scale_frame,
                      text = worst_label).pack(side = tkinter.LEFT, padx = 5)

        scale_value = tkinter.IntVar()

        radiobuttons = []
        
        for i in range(count):
            
            r = tkinter.Radiobutton(master = rating_scale_frame,
                                    text = "",
                                    variable = scale_value,
                                    value = i)

            r.pack(side = tkinter.LEFT, padx = 5)

            radiobuttons.append(r)

        # Set center button active
        # Note: .select() is not present in ttk.Radiobuttin
        #
        radiobuttons[int(len(radiobuttons) / 2)].invoke()
        
        tkinter.Label(master = rating_scale_frame,
                      text = best_label).pack(side = tkinter.LEFT, padx = 5)

        rating_scale_frame.pack()

        # IntVar() has a get() method, which is what we want to supply
        #
        return scale_value
def main():
    GUI.run('')

if __name__ == '__main__':
    main()
