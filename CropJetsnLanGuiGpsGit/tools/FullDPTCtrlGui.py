#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
import sys
import threading
# import tkinter
# import tkinter.ttk as ttk
if sys.version_info[0] == 2:  # Just checking your Python version to import Tkinter properly.
    from Tkinter import *
else:
    from tkinter import *
    from tkinter import ttk

from PIL import ImageTk, Image

import numpy as np
import pyzed.sl as sl
import cv2
from .zedcvstream import zedcv # import zed processing methods
from .CSysParams import CLanesParams,CAlgrthmParams,CCamsParams
import matplotlib.pyplot as plt
import threading
# from tools.LaneDeteTrack import CLneDetsTracks
# from tools.FullRGBLaneDeteTrack import CFullRGBLneDetsTracks
from tools.FullDptLaneDeteTrack import CFullDpthLneDetsTracks
from argparse import ArgumentParser as ArgParse
import os
# from  tools.sysmain import arwacmain

path = "/home/dom/ARWAC/data/ZED/"
# path = "/home/dom/ARWAC/ZED/"
# count_save = 0
# mode_point_cloud = 0
# mode_depth = 0
# point_cloud_format = sl.POINT_CLOUD_FORMAT.POINT_CLOUD_FORMAT_XYZ_ASCII
# depth_format = sl.DEPTH_FORMAT.DEPTH_FORMAT_PNG
m_CLanParams = object()
m_CAgthmParams = object()
m_CamParams  = object()
# m_CLneDTrcks = object()
# m_CFRGBLneTrcks = object()
m_CFDptLneTrcks = object()

class GUI(object):
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
    def __init__(self, master, **kwargs):#__init__(self, title = "Lanes Tracking GUI", width = 800):
        """Initialise. Call this first before calling any other tkinter routines.
           title is the window title to use.
           width is the width of scale, bar etc. widgets.
        """
        self.root = master#Tk()
        self.width =800
        self.root.title("Lane Configuring Console-Depth")
        self.root.attributes('-zoomed', True)  # This just maximizes it so we can see the window. It's nothing to do with fullscreen.
        self.frame = Frame(self.root)
        self.frame.pack()
        self.state = False
        self.root.bind("<Escape>", self.toggle_fullscreen)
        self.root.bind("<F11>", self.end_fullscreen)
        self.root.config(bg="skyblue")
        self.root.title("Lane Configuring Console")
        self.style = ttk.Style()
        self.scalwidth = 200
        self.threads = []
        self.exit = False
        self.initUI() # initializing the console - 
        
        self.m_OffsetCtrlBar = True # control offset increase or decrease
        self.m_LaneSetCtrlBar = True # control of initial number of lane setings, true - increase ++
        self.m_LaneOriSetCtrlBar = True # control of initial starting poiits of  left lane setings, true - increase ++
        self.m_LaneBandwithCtrlBar = True
        self.m_minLineLengthCtrlBar = True
        self.m_maxGapCtrlBar = True
        
        self.m_greenLowclorCtrlBar = True
        self.m_greenHigclorCtrlBar = True       
        self.m_greenLowSaturCtrlBar = True
        self.m_greenLowValCtrlBar = True    

        self.m_FullRGBRow0CtrlBar = True # control the initaila cropping position
        self.m_FullRGBH0CtrlBar=True
        self.m_FullRGBCol0CtrlBar = True
        self.m_FullRGBW0CtrlBar=True   



    def toggle_fullscreen(self, event=None):
        self.state = not self.state  # Just toggling the boolean
        self.root.attributes("-fullscreen", self.state)
        return "break"

    def end_fullscreen(self, event=None):
        self.state = False
        self.root.attributes("-fullscreen", False)
        return "break"

    def initUI(self, event=None):

        global m_CLanParams
        # global m_CAgthmParams
        # global m_CamParams 
        # global m_CLneDTrcks
        # global m_CFRGBLneTrcks
        # global m_CFDptLneTrcks
    
        m_CLanParams = CLanesParams()

         # Create left and right frames
        self.l_frm_wdth = 960
        self.l_frm_hght = 640
        self.r_frm_wdth = 480
        self.r_frm_hght = 320

        self.left_frame = Frame(self.root, width=self.l_frm_wdth, height=self.l_frm_hght, bg='grey')
        self.left_frame.pack(side='left', fill='both', padx=4, pady=4, expand=True)

        self.right_frame = Frame(self.root, width=self.r_frm_wdth, height=self.r_frm_hght, bg='grey')
        self.right_frame.pack(side='right', fill='both', padx=4, pady=4, expand=True)
        ####################################################
        # self.statusbar = Label(master = self.right_frame,text = "Ctrl.gui (tkinter {0})".format(TkVersion))
        # self.statusbar.pack(padx = 1, pady = 1)
        ####################################################

        """
        img = cv2.imread("path/to/img.png")

        # You may need to convert the color.
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        im_pil = Image.fromarray(img)

        # For reversing the operation:
        im_np = np.asarray(im_pil)
        opencv_image=cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR) 

        """

        # right_frame = Frame(root, width=650, height=400, bg='grey')
        # right_frame.pack(side='right', fill='both', padx=10, pady=5, expand=True)
        self.image_l=cv2.imread("./InitImages/20190516_142630.jpg")
        self.height_l=self.l_frm_hght
        self.width_l = self.l_frm_wdth
        
        self.size_l = (self.width_l,self.height_l)
        img = cv2.resize(self.image_l,self.size_l, interpolation = cv2.INTER_AREA)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.image_l = Image.fromarray(img)

        # # You may need to convert the color.
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # # For reversing the operation:
        # im_np = np.asarray(im_pil)
        # opencv_image=cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR) 
        # # self.width, self.height = self.image_l.size
        # self.height, self.width = self.image_l.size

        
        # self.image_l.thumbnail(size_l, Image.ANTIALIAS)
        # self.l_frm_wdth = 960
        # self.l_frm_hght = 640
        # self.r_frm_wdth = 256
        # self.r_frm_hght = 640
        Label(self.left_frame, text="Output Results Processed").pack(side='top', padx=1, pady=1)  
        # self.image_l = self.image_l.resize((self.height_l, self.width_l), Image.ANTIALIAS) ## The (250, 250) is (height, width)
        self.image_l = ImageTk.PhotoImage(self.image_l)
        # Label(self.left_frame, image=self.image_l).pack(fill='both', padx=4, pady=4)
        self.canvas_l = Canvas(self.left_frame, width = self.width_l, height = self.height_l)  
        self.canvas_l.pack()  
        self.canvas_l.create_image(4, 4, anchor=NW, image=self.image_l) 
        #####################################################

             
        #large_image = original_image.subsample(2,2)
        # self.height_r=int(self.height*0.05)
        # self.width_r = int(self.width*0.05)
        self.image_r=cv2.imread("./InitImages/20190516_142759.jpg")
  

        # size_r = [self.l_frm_wdth,self.l_frm_hght]
        self.height_r=int(self.r_frm_hght*0.5)
        self.width_r =int(self.r_frm_wdth)
        self.size_r= (self.width_r,self.height_r)
        img = cv2.resize(self.image_r,self.size_r, interpolation = cv2.INTER_AREA)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.image_r = Image.fromarray(img)
        # self.image_r.thumbnail(size_r, Image.ANTIALIAS)

        # self.image_r = self.image_r.resize((self.height_r, self.width_r), Image.ANTIALIAS) ## The (250, 250) is (height, width)
        Label(self.right_frame, text="Input Stream").pack(side='top', padx=1, pady=1) 
        self.image_r = ImageTk.PhotoImage(self.image_r)
        # Label(self.right_frame, image=self.image_r).pack(fill='both', padx=5, pady=5)
        self.canvas_r = Canvas(self.right_frame, width = self.width_r, height = self.height_r)  
        self.canvas_r.pack()  
        self.canvas_r.create_image(4, 4, anchor=NW, image=self.image_r)    

        """
        # Create frames and labels in left_frame
        Label(self.left_frame, text="Output Stream").pack(side='top', padx=4, pady=4)
        self.image_l = self.image.resize((int(self.height*0.5), int(self.width*0.5)), Image.ANTIALIAS) ## The (250, 250) is (height, width)

        self.image_l = ImageTk.PhotoImage(self.image_l)
        Label(self.left_frame, image=self.image_l).pack(fill='both', padx=4, pady=4)

        # canvas = Canvas(left_frame, width = width, height = height)  
        # canvas.pack()  
        # canvas.create_image(20, 20, anchor=NW, image=image_l) 

        Label(self.right_frame, text="Input Stream").pack(side='top', padx=5, pady=5)
        # Label(left_frame, image=image_l).pack(fill='both', padx=5, pady=5)
        #large_image = original_image.subsample(2,2)
        self.image_r = self.image.resize((int(self.height*0.05), int(self.width*0.05)), Image.ANTIALIAS) ## The (250, 250) is (height, width)
        self.image_r = ImageTk.PhotoImage(self.image_r)
        Label(self.right_frame, image=self.image_r).pack(fill='both', padx=5, pady=5)
        """

        self.tool_bar = Frame(self.right_frame, width=90, height=185, bg='lightgrey')
        self.tool_bar.pack(side='right', fill='both', padx=1, pady=1, expand=True)

        self.filter_bar = Frame(self.right_frame, width=90, height=185, bg='lightgrey')
        self.filter_bar.pack(side='left', fill='both',  padx=1, pady=1, expand=True)

        def clicked():
            '''if button is clicked, display message'''
            print("Clicked.")
        # Example labels that serve as placeholders for other widgets 
        Label(self.tool_bar, text="Tools", relief=RAISED).pack(anchor='n',  padx=1, pady=1, ipadx=1)
        Label(self.filter_bar, text="Filters", relief=RAISED).pack(anchor='n',  padx=1, pady=1, ipadx=1)

        # For now, when the buttons are clicked, they only call the clicked() method. We will add functionality later.
        # Button(self.tool_bar, text="Select", command=clicked).pack(padx=5, pady=5)
        # Button(self.tool_bar, text="Crop", command=clicked).pack(padx=5, pady=5)
        # Button(self.tool_bar, text="Rotate & Flip", command=clicked).pack( padx=1, pady=1)
        # Button(self.tool_bar, text="Resize", command=clicked).pack(padx=5, pady=5)
        # Button(self.filter_bar, text="Black & White", command=clicked).pack(padx=1, pady=1)
        """
        f = Frame(master=self.tool_bar, height=32, width=132)
        f.pack_propagate(0) # don't shrink
        f.pack()

        def ctr_msg():
            print("ctr_msg.....")
        b = Button(f, text="Sure!", command=ctr_msg)
        b.pack(fill=BOTH, expand=1)
        """
        # self.statusbar = Label(master = self.filter_bar,text = "Ctrl.gui (tkinter {0})".format(TkVersion))
        # self.statusbar.pack(padx = 1, pady = 1)
       
        return

    def add_statusbar(self):
        self.statusbar = Label(master = self.filter_bar,text = "Ctrl.gui (tkinter {0})".format(TkVersion))
        self.statusbar.pack(padx=1, pady=1)
        return

    def set_status(self, text):
        """Display the new status text in the status bar. """
        self.statusbar["text"] = text
        return
    #######################################

    def add_LaneSetstatusbar(self):
        strfm = 'default: ' + str(m_CLanParams.FullDTHNumRowsInit)
        self.statusbar2 = Label(master = self.filter_bar,text = "Lane number Setting {0})".format(strfm))
        self.statusbar2.pack(padx=1, pady=1)
        return    

    def set_LaneSetstatus(self, text):
        """Display the new status text in the status bar. """
        self.statusbar2["text"] = text
        return

    #########################################

    def add_LaneOriPtstatusbar(self):
        strfm = 'default: ' + str(m_CLanParams.FullDEPLaneOriPoint)
        self.statusbar3 = Label(master = self.filter_bar,text = "Lane Left Starting Poits Setting {0})".format(strfm))
        self.statusbar3.pack(padx=1, pady=1)
        return    

    def set_LaneOriPtstatus(self, text):
        """Display the new status text in the status bar. """
        self.statusbar3["text"] = text
        return

    ##########################################

    def add_LaneBandwithstatusbar(self):
        strfm = 'default: ' + str(m_CLanParams.FullDptbandwith)
        self.statusbar4 = Label(master = self.filter_bar,text = "Lane Bandwith Setting {0})".format(strfm))
        self.statusbar4.pack(padx=1, pady=1)
        return    

    def set_LaneBandwithstatus(self, text):
        """Display the new status text in the status bar. """
        self.statusbar4["text"] = text
        return

    ##########################################

    def add_minLineLengthstatusbar(self):
        strfm = 'default: ' + str(m_CLanParams.minLineLength_dpt)
        self.statusbar5 = Label(master = self.filter_bar,text = "HT minLineLength Setting {0})".format(strfm))
        self.statusbar5.pack(padx=1, pady=1)
        return    

    def set_minLineLengthstatus(self, text):
        """Display the new status text in the status bar. """
        self.statusbar5["text"] = text
        return

    ##########################################
    def add_maxGapstatusbar(self):
        strfm = 'default: ' + str(m_CLanParams.maxLineGap_dpt)
        self.statusbar6 = Label(master = self.filter_bar,text = "HT maxGap Setting {0})".format(strfm))
        self.statusbar6.pack(padx=1, pady=1)
        return    

    def set_maxGapstatus(self, text):
        """Display the new status text in the status bar. """
        self.statusbar6["text"] = text
        return

    ##########################################


    def add_greenLowclorstatusbar(self):
        strfm = 'default: ' + str(m_CLanParams.HSVLowGreenClor)
        self.statusbar7 = Label(master = self.filter_bar,text = "Lowband Color Setting {0})".format(strfm))
        self.statusbar7.pack(padx=1, pady=1)
        return    

    def set_greenLowclorstatus(self, text):
        """Display the new status text in the status bar. """
        self.statusbar7["text"] = text
        return
    ##########################################
    def add_greenHigclorstatusbar(self):
        strfm = 'default: ' + str(m_CLanParams.HSVHighGreenClor)
        # print('strfm',strfm)
        self.statusbar8 = Label(master = self.filter_bar,text = "Higband Color Setting {0})".format(strfm))
        self.statusbar8.pack(padx=1, pady=1)
        return    

    def set_greenHigclorstatus(self, text):
        """Display the new status text in the status bar. """
        self.statusbar8["text"] = text
        return
    ##########################################

    def add_greenLowSaturstatusbar(self):
        strfm = 'default: ' + str(m_CLanParams.HSVLowGreenSatur)
        # print('strfm',strfm)
        self.statusbar9 = Label(master = self.filter_bar,text = "Low Satur Setting {0})".format(strfm))
        self.statusbar9.pack(padx=1, pady=1)
        return    

    def set_greenLowSaturstatus(self, text):
        """Display the new status text in the status bar. """  
        self.statusbar9["text"] = text
        return

    ##########################################
    def add_greenLowValstatusbar(self):
        strfm = 'default: ' + str(m_CLanParams.HSVLowGreenVal)
        # print('strfm',strfm)
        self.statusbar10 = Label(master = self.filter_bar,text = "Low Val Setting {0})".format(strfm))
        self.statusbar10.pack(padx=1, pady=1)
        return    

    def set_greenLowValstatus(self, text):
        """Display the new status text in the status bar. """   
        self.statusbar10["text"] = text
        return

    ##########################################

    def add_FullRGBRow0Statusbar(self):
        strfm = 'default: ' + str(m_CLanParams.FullDptRow0)
        # print('strfm',strfm)
        self.statusbar11 = Label(master = self.filter_bar,text = "Low Row Crop Pt Setting {0})".format(strfm))
        self.statusbar11.pack(padx=1, pady=1)
        return    

    def set_FullRGBRow0Status(self, text):
        """Display the new status text in the status bar. """
        self.statusbar11["text"] = text
        return

    ##########################################
    def add_FullRGBH0Statusbar(self):
        strfm = 'default: ' + str(m_CLanParams.FullDptH0)
        # print('strfm',strfm)
        self.statusbar12 = Label(master = self.filter_bar,text = "Cropping Height Setting {0})".format(strfm))
        self.statusbar12.pack(padx=1, pady=1)
        return    

    def set_FullRGBH0Status(self, text):
        """Display the new status text in the status bar. """

        # print('statusbar8',text)
        self.statusbar12["text"] = text
        return

    ##########################################

    def add_FullRGBCol0Statusbar(self):
        strfm = 'default: ' + str(m_CLanParams.FullDptCol0)
        # print('strfm',strfm)
        self.statusbar13 = Label(master = self.filter_bar,text = "Low Col Crop Pt Setting {0})".format(strfm))
        self.statusbar13.pack(padx=1, pady=1)
        return    

    def set_FullRGBCol0Status(self, text):
        """Display the new status text in the status bar. """
        # print('statusbar8',text)
        self.statusbar13["text"] = text
        return

    ##########################################
    def add_FullRGBW0Statusbar(self):
        strfm = 'default: ' + str(m_CLanParams.FullDptW0)
        # print('strfm',strfm)
        self.statusbar14 = Label(master = self.filter_bar,text = "Cropping Width Setting {0})".format(strfm))
        self.statusbar14.pack(padx=1, pady=1)
        return    

    def set_FullRGBW0Status(self, text):
        """Display the new status text in the status bar. """

        # print('statusbar8',text)
        self.statusbar14["text"] = text
        return

    ##########################################

    def center(self):
        """Center the root window on screen.
        """
        self.right_frame.update()
        screenwidth = self.right_frame.winfo_screenwidth()
        screenheight = self.right_frame.winfo_screenheight()
        windowwidth = self.right_frame.winfo_width()
        windowheight = self.right_frame.winfo_height()
        self.right_frame.geometry("+{0}+{1}".format(int(screenwidth / 2 - windowwidth / 2),
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
            sys.stderr.write("Joining controls '" + thread.name + "'" + "\n")
            thread.join()

        sys.stderr.write("All possible threads joined" + "\n")
        # [In Python using Tkinter, what is the difference between root.destroy() and root.quit()?](https://stackoverflow.com/questions/2307464/in-python-using-tkinter-what-is-the-difference-between-root-destroy-and-root)
        #
        
        self.root.destroy()
        self.stop_zed()
        return

    def get_threaded(self, callback):
        """Return a function that, when called, will run `callback` in a background thread.
           The thread will be registrered in GUI.threads upon start.   """
        def threaded_callback():
            # Create a new Thread upon each call, so each Thread's
            # start() method is only called once.
            callback_thread = threading.Thread(target = callback,name = "Thread: " + callback.__name__)
            self.threads.append(callback_thread)
            # This will return immediately.
            callback_thread.start()
            return
        return threaded_callback

    def add_button(self, label, callback):
        """Add a button with label `label` calling `callback` with no arguments when clicked.
           The callback will run in a background thread.
        """
        # f = Frame(master=self.tool_bar, height=32, width=132)
        # f.pack_propagate(0) # don't shrink
        # f.pack()
        # def ctr_msg():
        #     print(label)
        # b = Button(f, text=label, command=ctr_msg)
        # b.pack(fill=BOTH, expand=1)
        Button(master = self.tool_bar, text = label, command = self.get_threaded(callback)).pack(padx = 1, pady = 1)
        return

    def SetStart_Ctrl(self):
        """Add a button with label `label` calling `callback` with no arguments when clicked.
           The callback will run in a background thread.
        """ 
        global m_CFDptLneTrcks
        m_CFDptLneTrcks.AgthmParams.ToIntializeSetting=False
        
        return


    def add_button2(self, label, callback):
        """Add a button with label `label` calling `callback` with no arguments when clicked.
           The callback will run in a background thread.
        """ 

        Button(master = self.filter_bar, text = label, command = self.get_threaded(callback)).pack(padx = 1, pady = 10)
        return

    def add_listbox(self, stringlist, callback):
        """Add a listbox.
           Klicking an entry in listbox will call `callback(entry)`,
           where `entry` is a string.
        """

        # TODO: What if the first item is the one to be selected, and no click happens? 
        # Wouldn't it be better to return something to query the selected item from, as done in labelentry?
        stringvar = StringVar(value = tuple(stringlist))
        listbox = Listbox(master = self.filter_bar,height = len(stringlist),listvariable = stringvar)

        listbox.pack(padx=1, pady=1)

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
        self.add_label(scalelabel)

        # No showvalue option for ttk Scale     
        value_label = Label(master = self.filter_bar,text = "0")
        value_label.pack(padx=1, pady=1)
        # TODO: Make callback threaded
        def callbackwrapper(value):
            # ttk.Scale calls back with a float string
            #
            value_label["text"] = int(float(value) * 10)
            callback(float(value))

        # self.style.configure("TScale",orient = "horizontal",sliderlength = 10)
        self.style.configure("TScale",orient = "vertical",sliderlength = 10)
        # self.style.configure("TScale", padding=6, relief="flat",background="#ccc")
        self.scalwidth = int(self.scalwidth*0.5)
        # Scale(master = self.right_frame,length = self.scalwidth,command = callbackwrapper).pack(padx = 4, pady = 4)
        w = Scale(master= self.filter_bar, from_=10, to=-10,command = callbackwrapper)
        w.pack(padx=1, pady=1)    
        return

    def add_label(self, text):
        """Add a label.
        """
        Label(master = self.filter_bar,text = text).pack(padx=1, pady=1)
        # Label(master = self.right_frame,text = text).pack(padx = 10, pady = 5)
        return

    def add_ImgOffsetBar(self):
        """Add a bar for progress or meter display.
           Returns a function to be called with a float 0..1.
       """
        global m_CLanParams
        # global m_CAgthmParams
        # global m_CamParams 
        llbar = int(self.width*0.5)
        barvaule = m_CLanParams.FullDEPOffset
        bar = ttk.Progressbar(master = self.filter_bar,
                                          orient = "horizontal",
                                          length = llbar,
                                          mode = "determinate",
                                          maximum = 200,
                                          value = barvaule)

        bar.pack(padx=1, pady=1)
        # self.set_status("Image OffSet(<200 pixels): '{0:.2}'".format(str(m_CLanParams.FullRGBoffset)))
        def callback(value):
            # self.SIMPOffset=75   # prospecitve transform
            # self.FullRGBoffset=25   # prospecitve transform
            # self.FullDEPoffset=25#25
            barvaule = int(value)
            # print('barvaule = ',barvaule)
            if self.m_OffsetCtrlBar == True:
                # m_CLanParams.FullDEPoffset = m_CLanParams.SIMPOffset + barvaule
                # m_CLanParams.FullRGBoffset = m_CLanParams.FullRGBoffset + barvaule
                m_CLanParams.FullDEPOffset = m_CLanParams.FullDEPOffset + barvaule
                bar["value"] = m_CLanParams.FullDEPOffset#barvaule#int(value * 10)
                self.set_status("Image OffSet(<200 pixels): '{0:.3}'".format(str(m_CLanParams.FullDEPOffset)))
            else:
                # m_CLanParams.SIMPOffset = m_CLanParams.SIMPOffset - barvaule
                # m_CLanParams.FullRGBoffset = m_CLanParams.FullRGBoffset - barvaule
                m_CLanParams.FullDEPOffset = m_CLanParams.FullDEPOffset - barvaule
                bar["value"] = m_CLanParams.FullDEPOffset#barvaule#int(value * 10)
                self.set_status("Image OffSet(>0 pixels): '{0:.3}'".format(str(m_CLanParams.FullDEPOffset)))


            return
        return callback

    ########################################
    def add_LaneSettingBar(self):
        """Add a bar for progress or meter display.
           Returns a function to be called with a float 0..1.
        """
        global m_CLanParams
        # global m_CAgthmParams
        # global m_CamParams 
        llbar = int(self.width*0.5)
        barvaule = m_CLanParams.FullDTHNumRowsInit

        bar = ttk.Progressbar(master = self.filter_bar,
                                          orient = "horizontal",
                                          length = llbar,
                                          mode = "determinate",
                                          maximum = 20,
                                          value = barvaule)

        bar.pack(padx=1, pady=1)
        # self.set_status("Image OffSet(<200 pixels): '{0:.2}'".format(str(m_CLanParams.FullRGBoffset)))
        def callback(value):
            # self.SIMPOffset=75   # prospecitve transform
            # self.FullRGBoffset=25   # prospecitve transform
            # self.FullDEPoffset=25#25
            barvaule = int(value)
            # print('barvaule = ',barvaule)
            if self.m_LaneSetCtrlBar == True:
                m_CLanParams.FullDTHNumRowsInit = m_CLanParams.FullDTHNumRowsInit + barvaule
                # m_CLanParams.FullRGBNumRowsInit = m_CLanParams.FullRGBNumRowsInit + barvaule
                # m_CLanParams.FullRGBNumRowsInit = m_CLanParams.FullRGBNumRowsInit + barvaule
                bar["value"] = m_CLanParams.FullDTHNumRowsInit#barvaule#int(value * 10)
                self.set_LaneSetstatus("Total Lane Num < 20): '{0:.2}'".format(str(m_CLanParams.FullDTHNumRowsInit)))
            else:
                m_CLanParams.FullDTHNumRowsInit = m_CLanParams.FullDTHNumRowsInit - barvaule
                bar["value"] = m_CLanParams.FullDTHNumRowsInit#barvaule#int(value * 10)
                self.set_LaneSetstatus("Total Lane Num > 0): '{0:.2}'".format(str(m_CLanParams.FullDTHNumRowsInit)))

            return
        return callback        


    #########################################
    def add_LaneOriPtsBar(self):
        """Add a bar for progress or meter display.
           Returns a function to be called with a float 0..1.
        """
        global m_CLanParams
        # global m_CAgthmParams
        # global m_CamParams 
        llbar = int(self.width*0.5)
        barvaule = m_CLanParams.FullDEPLaneOriPoint
        bar = ttk.Progressbar(master = self.filter_bar,
                                          orient = "horizontal",
                                          length = llbar,
                                          mode = "determinate",
                                          maximum = 200,
                                          value = barvaule)

        bar.pack(padx=1, pady=1)
        # self.set_status("Image OffSet(<200 pixels): '{0:.2}'".format(str(m_CLanParams.FullRGBoffset)))
        def callback(value):
            # self.SIMPOffset=75   # prospecitve transform
            # self.FullRGBoffset=25   # prospecitve transform
            # self.FullDEPoffset=25#25
            barvaule = int(value)
            # print('barvaule = ',barvaule)
            if self.m_LaneOriSetCtrlBar == True:
                m_CLanParams.FullDEPLaneOriPoint = m_CLanParams.FullDEPLaneOriPoint + barvaule
                # m_CLanParams.FullRGBNumRowsInit = m_CLanParams.FullRGBNumRowsInit + barvaule
                # m_CLanParams.FullRGBNumRowsInit = m_CLanParams.FullRGBNumRowsInit + barvaule
                bar["value"] = m_CLanParams.FullDEPLaneOriPoint#barvaule#int(value * 10)
                self.set_LaneOriPtstatus("Lane Start Points < 200): '{0:.3}'".format(str(m_CLanParams.FullDEPLaneOriPoint)))
            else:
                m_CLanParams.FullDEPLaneOriPoint = m_CLanParams.FullDEPLaneOriPoint - barvaule
                bar["value"] = m_CLanParams.FullDEPLaneOriPoint#barvaule#int(value * 10)
                self.set_LaneOriPtstatus("Lane Start Points > 0): '{0:.3}'".format(str(m_CLanParams.FullDEPLaneOriPoint)))

            return
        return callback        
    ##############################################

    def add_LaneBandwithBar(self):
        """Add a bar for progress or meter display.
           Returns a function to be called with a float 0..1.
        """
        global m_CLanParams
        # global m_CAgthmParams
        # global m_CamParams 
        llbar = int(self.width*0.5)
        barvaule = m_CLanParams.FullDptbandwith
        bar = ttk.Progressbar(master = self.filter_bar,
                                          orient = "horizontal",
                                          length = llbar,
                                          mode = "determinate",
                                          maximum = 200,
                                          value = barvaule)

        bar.pack(padx=1, pady=1)
        # self.set_status("Image OffSet(<200 pixels): '{0:.2}'".format(str(m_CLanParams.FullRGBoffset)))
        def callback(value):
            # self.SIMPOffset=75   # prospecitve transform
            # self.FullRGBoffset=25   # prospecitve transform
            # self.FullDEPoffset=25#25
            barvaule = int(value)
            # print('barvaule = ',barvaule)
            if self.m_LaneBandwithCtrlBar == True:
                m_CLanParams.FullDptbandwith = m_CLanParams.FullDptbandwith + barvaule
                # m_CLanParams.FullRGBNumRowsInit = m_CLanParams.FullRGBNumRowsInit + barvaule
                # m_CLanParams.FullRGBNumRowsInit = m_CLanParams.FullRGBNumRowsInit + barvaule
                bar["value"] = m_CLanParams.FullDptbandwith#barvaule#int(value * 10)
                self.set_LaneBandwithstatus("Lane Bandwith  < 200): '{0:.3}'".format(str(m_CLanParams.FullDptbandwith)))
            else:
                m_CLanParams.FullDptbandwith = m_CLanParams.FullDptbandwith - barvaule
                bar["value"] = m_CLanParams.FullDptbandwith#barvaule#int(value * 10)
                self.set_LaneBandwithstatus("Lane Bandwith > 0): '{0:.3}'".format(str(m_CLanParams.FullDptbandwith)))

            return
        return callback     

    ##############################################

    # def add_LaneBandwithBar(self):
    #     """Add a bar for progress or meter display.
    #        Returns a function to be called with a float 0..1.
    #     """
    #     global m_CLanParams
    #     # global m_CAgthmParams
    #     # global m_CamParams 
    #     llbar = int(self.width*0.5)
    #     barvaule = m_CLanParams.FullRGBbandwith
    #     bar = ttk.Progressbar(master = self.filter_bar,
    #                                       orient = "horizontal",
    #                                       length = llbar,
    #                                       mode = "determinate",
    #                                       maximum = 200,
    #                                       value = barvaule)

    #     bar.pack(padx=1, pady=1)
    #     # self.set_status("Image OffSet(<200 pixels): '{0:.2}'".format(str(m_CLanParams.FullRGBoffset)))
    #     def callback(value):
    #         # self.SIMPOffset=75   # prospecitve transform
    #         # self.FullRGBoffset=25   # prospecitve transform
    #         # self.FullDEPoffset=25#25
    #         barvaule = int(value)
    #         # print('barvaule = ',barvaule)
    #         if self.m_LaneBandwithCtrlBar == True:
    #             m_CLanParams.FullRGBbandwith = m_CLanParams.FullRGBbandwith + barvaule
    #             # m_CLanParams.FullRGBNumRowsInit = m_CLanParams.FullRGBNumRowsInit + barvaule
    #             # m_CLanParams.FullRGBNumRowsInit = m_CLanParams.FullRGBNumRowsInit + barvaule
    #             bar["value"] = m_CLanParams.FullRGBbandwith#barvaule#int(value * 10)
    #             self.set_LaneBandwithstatus("Lane Bandwith  < 200): '{0:.3}'".format(str(m_CLanParams.FullRGBbandwith)))
    #         else:
    #             m_CLanParams.FullRGBbandwith = m_CLanParams.FullRGBbandwith - barvaule
    #             bar["value"] = m_CLanParams.FullRGBbandwith#barvaule#int(value * 10)
    #             self.set_LaneBandwithstatus("Lane Bandwith > 0): '{0:.3}'".format(str(m_CLanParams.FullRGBbandwith)))

    #         return
    #     return callback     


    ###############################################

    def add_minLineLengthBar(self):
        """Add a bar for progress or meter display.
           Returns a function to be called with a float 0..1.
        """
        global m_CLanParams
        # global m_CAgthmParams
        # global m_CamParams 
        llbar = int(self.width*0.5)
        barvaule = m_CLanParams.minLineLength_dpt
        bar = ttk.Progressbar(master = self.filter_bar,
                                          orient = "horizontal",
                                          length = llbar,
                                          mode = "determinate",
                                          maximum = 200,
                                          value = barvaule)

        bar.pack(padx=1, pady=1)
        # self.set_status("Image OffSet(<200 pixels): '{0:.2}'".format(str(m_CLanParams.FullRGBoffset)))
        def callback(value):
            # self.SIMPOffset=75   # prospecitve transform
            # self.FullRGBoffset=25   # prospecitve transform
            # self.FullDEPoffset=25#25
            barvaule = int(value)
            # print('barvaule = ',barvaule)
            if self.m_minLineLengthCtrlBar == True:
                m_CLanParams.minLineLength_dpt = m_CLanParams.minLineLength_dpt + barvaule
                # m_CLanParams.FullRGBNumRowsInit = m_CLanParams.FullRGBNumRowsInit + barvaule
                # m_CLanParams.FullRGBNumRowsInit = m_CLanParams.FullRGBNumRowsInit + barvaule
                bar["value"] = m_CLanParams.minLineLength_dpt#barvaule#int(value * 10)
                self.set_minLineLengthstatus("minLineLength  < 200): '{0:.3}'".format(str(m_CLanParams.minLineLength_dpt)))
            else:
                m_CLanParams.minLineLength_dpt = m_CLanParams.minLineLength_dpt - barvaule
                bar["value"] = m_CLanParams.minLineLength_dpt#barvaule#int(value * 10)
                self.set_minLineLengthstatus("minLineLength > 0): '{0:.3}'".format(str(m_CLanParams.minLineLength_dpt)))

            return
        return callback     
    ###############################################

    def add_maxGapBar(self):
        """Add a bar for progress or meter display.
           Returns a function to be called with a float 0..1.
        """
        global m_CLanParams
        # global m_CAgthmParams
        # global m_CamParams 
        llbar = int(self.width*0.5)
        barvaule = m_CLanParams.maxLineGap_dpt
        bar = ttk.Progressbar(master = self.filter_bar,
                                          orient = "horizontal",
                                          length = llbar,
                                          mode = "determinate",
                                          maximum = 200,
                                          value = barvaule)

        bar.pack(padx=1, pady=1)
        # self.set_status("Image OffSet(<200 pixels): '{0:.2}'".format(str(m_CLanParams.FullRGBoffset)))
        def callback(value):
            # self.SIMPOffset=75   # prospecitve transform
            # self.FullRGBoffset=25   # prospecitve transform
            # self.FullDEPoffset=25#25
            barvaule = int(value)
            # print('barvaule = ',barvaule)
            if self.m_maxGapCtrlBar == True:
                m_CLanParams.maxLineGap_dpt = m_CLanParams.maxLineGap_dpt + barvaule
                # m_CLanParams.FullRGBNumRowsInit = m_CLanParams.FullRGBNumRowsInit + barvaule
                # m_CLanParams.FullRGBNumRowsInit = m_CLanParams.FullRGBNumRowsInit + barvaule
                bar["value"] = m_CLanParams.maxLineGap_dpt#barvaule#int(value * 10)
                self.set_maxGapstatus("maxLineGap  < 200): '{0:.3}'".format(str(m_CLanParams.maxLineGap_dpt)))
            else:
                m_CLanParams.maxLineGap = m_CLanParams.maxLineGap_dpt - barvaule
                bar["value"] = m_CLanParams.maxLineGap_dpt#barvaule#int(value * 10)
                self.set_maxGapstatus("maxLineGap > 0): '{0:.3}'".format(str(m_CLanParams.maxLineGap_dpt)))

            return
        return callback     



    ###############################################

    def add_greenLowclorBar(self):
            """Add a bar for progress or meter display.
            Returns a function to be called with a float 0..1.
            """
            global m_CLanParams
            # global m_CAgthmParams
            # global m_CamParams 
            llbar = int(self.width*0.5)
            barvaule = m_CLanParams.HSVLowGreenClor
            bar = ttk.Progressbar(master = self.filter_bar,
                                            orient = "horizontal",
                                            length = llbar,
                                            mode = "determinate",
                                            maximum = 180,
                                            value = barvaule)

            bar.pack(padx=1, pady=1)
            # self.set_status("Image OffSet(<200 pixels): '{0:.2}'".format(str(m_CLanParams.FullRGBoffset)))
            def callback(value):
                # self.SIMPOffset=75   # prospecitve transform
                # self.FullRGBoffset=25   # prospecitve transform
                # self.FullDEPoffset=25#25
                barvaule = int(value)
                # print('barvaule = ',barvaule)
                if self.m_greenLowclorCtrlBar == True:
                    m_CLanParams.HSVLowGreenClor = m_CLanParams.HSVLowGreenClor + barvaule
                    # m_CLanParams.FullRGBNumRowsInit = m_CLanParams.FullRGBNumRowsInit + barvaule
                    # m_CLanParams.FullRGBNumRowsInit = m_CLanParams.FullRGBNumRowsInit + barvaule
                    bar["value"] = m_CLanParams.HSVLowGreenClor#barvaule#int(value * 10)
                    self.set_greenLowclorstatus("HSVLowGreenClor  < 180): '{0:.3}'".format(str(m_CLanParams.HSVLowGreenClor)))
                else:
                    m_CLanParams.HSVLowGreenClor = m_CLanParams.HSVLowGreenClor - barvaule
                    bar["value"] = m_CLanParams.HSVLowGreenClor#barvaule#int(value * 10)
                    self.set_greenLowclorstatus("HSVLowGreenClor > 25): '{0:.3}'".format(str(m_CLanParams.HSVLowGreenClor)))

                return
            return callback     

    ###############################################

    def add_greenHigclorBar(self):
            """Add a bar for progress or meter display.
            Returns a function to be called with a float 0..1.
            """
            global m_CLanParams
            # global m_CAgthmParams
            # global m_CamParams 
            llbar = int(self.width*0.5)
            barvaule = m_CLanParams.HSVHighGreenClor
            bar = ttk.Progressbar(master = self.filter_bar,
                                            orient = "horizontal",
                                            length = llbar,
                                            mode = "determinate",
                                            maximum = 180,
                                            value = barvaule)

            bar.pack(padx=1, pady=1)
            # self.set_status("Image OffSet(<200 pixels): '{0:.2}'".format(str(m_CLanParams.FullRGBoffset)))
            def callback(value):
                # self.SIMPOffset=75   # prospecitve transform
                # self.FullRGBoffset=25   # prospecitve transform
                # self.FullDEPoffset=25#25
                barvaule = int(value)
                # print('barvaule = ',barvaule)
                if self.m_greenHigclorCtrlBar == True:
                    m_CLanParams.HSVHighGreenClor = m_CLanParams.HSVHighGreenClor + barvaule
                    # m_CLanParams.FullRGBNumRowsInit = m_CLanParams.FullRGBNumRowsInit + barvaule
                    # m_CLanParams.FullRGBNumRowsInit = m_CLanParams.FullRGBNumRowsInit + barvaule
                    bar["value"] = m_CLanParams.HSVHighGreenClor#barvaule#int(value * 10)
                    self.set_greenHigclorstatus("HSVHigGreenClor  < 180): '{0:.3}'".format(str(m_CLanParams.HSVHighGreenClor)))
                else:
                    m_CLanParams.HSVHighGreenClor = m_CLanParams.HSVHighGreenClor - barvaule
                    bar["value"] = m_CLanParams.HSVHighGreenClor#barvaule#int(value * 10)
                    # print('m_CLanParams.HSVHighGreenClor=',m_CLanParams.HSVHighGreenClor)
                    self.set_greenHigclorstatus("HSVHigGreenClor > 25): '{0:.3}'".format(str(m_CLanParams.HSVHighGreenClor)))

                return
            return callback    

    ###############################################
    def add_greenLowSaturBar(self):
            """Add a bar for progress or meter display.
            Returns a function to be called with a float 0..1.
            """
            global m_CLanParams
            # global m_CAgthmParams
            # global m_CamParams 
            llbar = int(self.width*0.5)
            barvaule = m_CLanParams.HSVLowGreenSatur
            bar = ttk.Progressbar(master = self.filter_bar,
                                            orient = "horizontal",
                                            length = llbar,
                                            mode = "determinate",
                                            maximum = 225,
                                            value = barvaule)

            bar.pack(padx=1, pady=1)
            # self.set_status("Image OffSet(<200 pixels): '{0:.2}'".format(str(m_CLanParams.FullRGBoffset)))
            def callback(value):
                # self.SIMPOffset=75   # prospecitve transform
                # self.FullRGBoffset=25   # prospecitve transform
                # self.FullDEPoffset=25#25
                barvaule = int(value)
                # print('barvaule = ',barvaule)
                if self.m_greenLowSaturCtrlBar == True:
                    m_CLanParams.HSVLowGreenSatur = m_CLanParams.HSVLowGreenSatur + barvaule
                    # m_CLanParams.FullRGBNumRowsInit = m_CLanParams.FullRGBNumRowsInit + barvaule
                    # m_CLanParams.FullRGBNumRowsInit = m_CLanParams.FullRGBNumRowsInit + barvaule
                    bar["value"] = m_CLanParams.HSVLowGreenSatur#barvaule#int(value * 10)
                    self.set_greenLowSaturstatus("LowSatur  < 225): '{0:.3}'".format(str(m_CLanParams.HSVLowGreenSatur)))
                else:
                    m_CLanParams.HSVLowGreenSatur = m_CLanParams.HSVLowGreenSatur - barvaule
                    bar["value"] = m_CLanParams.HSVLowGreenSatur#barvaule#int(value * 10)
                    # print('m_CLanParams.HSVHighGreenClor=',m_CLanParams.HSVHighGreenClor)
                    self.set_greenLowSaturstatus("LowSatur > 25): '{0:.3}'".format(str(m_CLanParams.HSVLowGreenSatur)))

                return
            return callback    

    ###############################################
    def add_greenLowValBar(self):
            """Add a bar for progress or meter display.
            Returns a function to be called with a float 0..1.
            """
            global m_CLanParams
            # global m_CAgthmParams
            # global m_CamParams 
            llbar = int(self.width*0.5)
            barvaule = m_CLanParams.HSVLowGreenVal
            bar = ttk.Progressbar(master = self.filter_bar,
                                            orient = "horizontal",
                                            length = llbar,
                                            mode = "determinate",
                                            maximum = 225,
                                            value = barvaule)

            bar.pack(padx=1, pady=1)
            # self.set_status("Image OffSet(<200 pixels): '{0:.2}'".format(str(m_CLanParams.FullRGBoffset)))
            def callback(value):
                # self.SIMPOffset=75   # prospecitve transform
                # self.FullRGBoffset=25   # prospecitve transform
                # self.FullDEPoffset=25#25
                barvaule = int(value)
                # print('barvaule = ',barvaule)
                if self.m_greenLowValCtrlBar == True:
                    m_CLanParams.HSVLowGreenVal = m_CLanParams.HSVLowGreenVal + barvaule
                    # m_CLanParams.FullRGBNumRowsInit = m_CLanParams.FullRGBNumRowsInit + barvaule
                    # m_CLanParams.FullRGBNumRowsInit = m_CLanParams.FullRGBNumRowsInit + barvaule
                    bar["value"] = m_CLanParams.HSVLowGreenVal#barvaule#int(value * 10)
                    self.set_greenLowValstatus("HSVLowVal  < 225): '{0:.3}'".format(str(m_CLanParams.HSVLowGreenVal)))
                else:
                    m_CLanParams.HSVLowGreenVal = m_CLanParams.HSVLowGreenVal - barvaule
                    bar["value"] = m_CLanParams.HSVLowGreenVal#barvaule#int(value * 10)
                    # print('m_CLanParams.HSVLowGreenVal=',m_CLanParams.HSVLowGreenVal)
                    self.set_greenLowValstatus("HSVLowVal > 25): '{0:.3}'".format(str(m_CLanParams.HSVLowGreenVal)))

                return
            return callback    

    ###############################################
    def add_FullRGBRow0Bar(self):
            """Add a bar for progress or meter display.
            Returns a function to be called with a float 0..1.
            """
            global m_CLanParams
            # global m_CAgthmParams
            # global m_CamParams 
            llbar = int(self.width*0.5)
            barvaule = m_CLanParams.FullDptRow0
            bar = ttk.Progressbar(master = self.filter_bar,
                                            orient = "horizontal",
                                            length = llbar,
                                            mode = "determinate",
                                            maximum = 500,
                                            value = barvaule)

            bar.pack(padx=1, pady=1)
            # self.set_status("Image OffSet(<200 pixels): '{0:.2}'".format(str(m_CLanParams.FullRGBoffset)))
            def callback(value):
                # self.SIMPOffset=75   # prospecitve transform
                # self.FullRGBoffset=25   # prospecitve transform
                # self.FullDEPoffset=25#25
                barvaule = int(value)
                # print('barvaule = ',barvaule)
                if self.m_FullRGBRow0CtrlBar == True:
                    m_CLanParams.FullDptRow0 = m_CLanParams.FullDptRow0 + barvaule
                    # m_CLanParams.FullRGBNumRowsInit = m_CLanParams.FullRGBNumRowsInit + barvaule
                    # m_CLanParams.FullRGBNumRowsInit = m_CLanParams.FullRGBNumRowsInit + barvaule
                    bar["value"] = m_CLanParams.FullDptRow0#barvaule#int(value * 10)
                    self.set_FullRGBRow0Status("FullDptRow0  < 500): '{0:.3}'".format(str(m_CLanParams.FullDptRow0)))
                else:
                    m_CLanParams.FullDptRow0 = m_CLanParams.FullDptRow0 - barvaule
                    bar["value"] = m_CLanParams.FullDptRow0#barvaule#int(value * 10)
                    # print('m_CLanParams.HSVLowGreenVal=',m_CLanParams.HSVLowGreenVal)
                    self.set_FullRGBRow0Status("FullDptRow0 >= 0): '{0:.3}'".format(str(m_CLanParams.FullDptRow0)))

                return
            return callback   
    ###############################################

    def add_FullRGBH0Bar(self):
            """Add a bar for progress or meter display.
            Returns a function to be called with a float 0..1.
            """
            global m_CLanParams
            # global m_CAgthmParams
            # global m_CamParams 
            llbar = int(self.width*0.5)
            barvaule = m_CLanParams.FullDptH0
            bar = ttk.Progressbar(master = self.filter_bar,
                                            orient = "horizontal",
                                            length = llbar,
                                            mode = "determinate",
                                            maximum = 1960,
                                            value = barvaule)

            bar.pack(padx=1, pady=1)
            # self.set_status("Image OffSet(<200 pixels): '{0:.2}'".format(str(m_CLanParams.FullRGBoffset)))
            def callback(value):
                # self.SIMPOffset=75   # prospecitve transform
                # self.FullRGBoffset=25   # prospecitve transform
                # self.FullDEPoffset=25#25
                barvaule = int(value)
                # print('barvaule = ',barvaule)
                if self.m_FullRGBH0CtrlBar == True:
                    m_CLanParams.FullDptH0 = m_CLanParams.FullDptH0 + barvaule
                    # m_CLanParams.FullRGBNumRowsInit = m_CLanParams.FullRGBNumRowsInit + barvaule
                    # m_CLanParams.FullRGBNumRowsInit = m_CLanParams.FullRGBNumRowsInit + barvaule
                    bar["value"] = m_CLanParams.FullDptH0#barvaule#int(value * 10)
                    self.set_FullRGBH0Status("FullDptH0  < 1960): '{0:.3}'".format(str(m_CLanParams.FullDptH0)))
                else:
                    m_CLanParams.FullDptH0 = m_CLanParams.FullDptH0 - barvaule
                    bar["value"] = m_CLanParams.FullDptH0#barvaule#int(value * 10)
                    # print('m_CLanParams.HSVLowGreenVal=',m_CLanParams.HSVLowGreenVal)
                    self.set_FullRGBH0Status("FullDptH0 >= 0): '{0:.3}'".format(str(m_CLanParams.FullDptH0)))

                return
            return callback    

    ###############################################


    def add_FullRGBCol0Bar(self):
            """Add a bar for progress or meter display.
            Returns a function to be called with a float 0..1.
            """
            global m_CLanParams
            # global m_CAgthmParams
            # global m_CamParams 
            llbar = int(self.width*0.5)
            barvaule = m_CLanParams.FullDptCol0
            bar = ttk.Progressbar(master = self.filter_bar,
                                            orient = "horizontal",
                                            length = llbar,
                                            mode = "determinate",
                                            maximum = 1960,
                                            value = barvaule)

            bar.pack(padx=1, pady=1)
            # self.set_status("Image OffSet(<200 pixels): '{0:.2}'".format(str(m_CLanParams.FullRGBoffset)))
            def callback(value):
                # self.SIMPOffset=75   # prospecitve transform
                # self.FullRGBoffset=25   # prospecitve transform
                # self.FullDEPoffset=25#25
                barvaule = int(value)
                # print('barvaule = ',barvaule)
                if self.m_FullRGBCol0CtrlBar == True:
                    m_CLanParams.FullDptCol0 = m_CLanParams.FullDptCol0 + barvaule
                    # m_CLanParams.FullRGBNumRowsInit = m_CLanParams.FullRGBNumRowsInit + barvaule
                    # m_CLanParams.FullRGBNumRowsInit = m_CLanParams.FullRGBNumRowsInit + barvaule
                    bar["value"] = m_CLanParams.FullDptCol0#barvaule#int(value * 10)
                    self.set_FullRGBCol0Status("FullDptCol0  < 1960): '{0:.3}'".format(str(m_CLanParams.FullDptCol0)))
                else:
                    m_CLanParams.FullDptCol0 = m_CLanParams.FullDptCol0 - barvaule
                    bar["value"] = m_CLanParams.FullDptCol0#barvaule#int(value * 10)
                    # print('m_CLanParams.HSVLowGreenVal=',m_CLanParams.HSVLowGreenVal)
                    self.set_FullRGBCol0Status("FullDptCol0 >= 0): '{0:.3}'".format(str(m_CLanParams.FullDptCol0)))

                return
            return callback    

    ###############################################
    def add_FullRGBW0Bar(self):
            """Add a bar for progress or meter display.
            Returns a function to be called with a float 0..1.
            """
            global m_CLanParams
            # global m_CAgthmParams
            # global m_CamParams 
            llbar = int(self.width*0.5)
            barvaule = m_CLanParams.FullDptW0
            bar = ttk.Progressbar(master = self.filter_bar,
                                            orient = "horizontal",
                                            length = llbar,
                                            mode = "determinate",
                                            maximum = 1960,
                                            value = barvaule)

            bar.pack(padx=1, pady=1)
            # self.set_status("Image OffSet(<200 pixels): '{0:.2}'".format(str(m_CLanParams.FullRGBoffset)))
            def callback(value):
                # self.SIMPOffset=75   # prospecitve transform
                # self.FullRGBoffset=25   # prospecitve transform
                # self.FullDEPoffset=25#25
                barvaule = int(value)
                # print('barvaule = ',barvaule)
                if self.m_FullRGBW0CtrlBar == True:
                    m_CLanParams.FullDptW0 = m_CLanParams.FullDptW0 + barvaule
                    # m_CLanParams.FullRGBNumRowsInit = m_CLanParams.FullRGBNumRowsInit + barvaule
                    # m_CLanParams.FullRGBNumRowsInit = m_CLanParams.FullRGBNumRowsInit + barvaule
                    bar["value"] = m_CLanParams.FullDptW0#barvaule#int(value * 10)
                    self.set_FullRGBW0Status("FullDptW0  < 1960): '{0:.3}'".format(str(m_CLanParams.FullDptW0)))
                else:
                    m_CLanParams.FullDptW0 = m_CLanParams.FullDptW0 - barvaule
                    bar["value"] = m_CLanParams.FullDptW0#barvaule#int(value * 10)
                    # print('m_CLanParams.HSVLowGreenVal=',m_CLanParams.HSVLowGreenVal)
                    self.set_FullRGBW0Status("FullDptW0 >= 0): '{0:.3}'".format(str(m_CLanParams.FullDptW0)))

                return
            return callback    

    ###############################################


    def add_canvas(self, width, height):
        #Add a canvas and return it. 
        canvas = Canvas(master = self.filter_bar, width = width, height = height)
        canvas.pack(padx=1, pady=1)
        return canvas
    
    def add_labelentry(self, text, width = 16, content = ""):
        """Create a labelled entry, and return the entry.
           width is the number of characters to show.
           content is a string to show in the entry.
           Call entry.get() to get its current content.
        """ 
        Label(master = self.filter_bar,text = "Data Input Below: ").pack(padx=1, pady=1)
        labelentry = Frame(master = self.filter_bar)
        Label(master = labelentry,text = text).pack(side = LEFT)

        entry = Entry(master = labelentry,width = width)
        entry.insert(0, content)
        entry.pack()
        labelentry.pack()
        return entry

    def add_rating_scale(self, scalelabel, count, worst_label, best_label):
        """Add a rating scale, with `count` options to choose from.
           Returns an object whose get() method will yield the integer
           index of the selected item, starting with 0.
        """
        Label(master = self.filter_bar,text = scalelabel).pack(padx=1, pady=1)

        rating_scale_frame = Frame(master = self.filter_bar)
        Label(master = rating_scale_frame,text = worst_label).pack(side = LEFT, padx =1)
        scale_value = IntVar()
        radiobuttons = []        
        for i in range(count):            
            r = Radiobutton(master = rating_scale_frame,
                                    text = "",
                                    variable = scale_value,
                                    value = 1)

            r.pack(side = LEFT, padx = 1)
            radiobuttons.append(r)

        # Set center button active
        # Note: .select() is not present in ttk.Radiobuttin
        #
        radiobuttons[int(len(radiobuttons) / 2)].invoke()  
        # radiobuttons[0].invoke()        
        Label(master = rating_scale_frame,
                      text = best_label).pack(side = LEFT, padx = 1)
        rating_scale_frame.pack()
        # IntVar() has a get() method, which is what we want to supply
        #
        return scale_value
    def startproces(self,ArgAll):
        
        self.zed = sl.Camera()
        if self.zed.is_opened()==True:
            self.zed.close()
        # Set configuration parameters
        init = sl.InitParameters()
        # init.camera_resolution = sl.RESOLUTION.RESOLUTION_HD1080
        # init.depth_mode = sl.DEPTH_MODE.DEPTH_MODE_ULTRA#.DEPTH_MODE_PERFORMANCE
        # init.coordinate_units = sl.UNIT.UNIT_METER
        # if len(sys.argv) >= 2 :
        #     init.svo_input_filename = sys.argv[1]
        input_type = sl.InputType()
        if len(sys.argv) >= 2 :
            input_type.set_from_svo_file(sys.argv[1])
            
        init = sl.InitParameters(input_t=input_type)
        init.camera_resolution = sl.RESOLUTION.HD720#.HD1080
        init.depth_mode = sl.DEPTH_MODE.PERFORMANCE
        init.coordinate_units = sl.UNIT.METER#MILLIMETER
        init.depth_minimum_distance=0.5#minimum depth perception distance to half meter
        init.depth_maximum_distance=20#maxium depth percetion distan to 20 meter
        init.camera_fps=30#set fps at 30    

        # init = sl.InitParameters()
        # init.camera_resolution = sl.RESOLUTION.RESOLUTION_HD1080
        # init.depth_mode = sl.DEPTH_MODE.DEPTH_MODE_ULTRA#.DEPTH_MODE_PERFORMANCE
        # init.coordinate_units = sl.UNIT.UNIT_METER
        # if len(sys.argv) >= 2 :
        #     init.svo_input_filename = sys.argv[1]

        # Open the camera
        # zed.close()
        err = self.zed.open(init)
        if err != sl.ERROR_CODE.SUCCESS :
            print(repr(err))
            self.zed.close()
            exit(1)

        
        global m_CLanParams
        global m_CAgthmParams
        global m_CamParams 
        # global m_CLneDTrcks
        # global m_CFRGBLneTrcks
        global m_CFDptLneTrcks
    
        # m_CLanParams = CLanesParams()
        m_CamParams = CCamsParams(self.zed, sl)


        # inputpth ="/home/chfox/ARWAC/Essa-0/20190206_143625.mp4"  
        video_ocv = "/home/dom/ARWAC/data/Input/VGA_SN12925_13-29-42_l.avi" 
        video_dpt = "/home/dom/ARWAC/data/Input/VGA_SN12925_13-29-42_d.avi"     
        outputpth ='/home/dom/Documents/ARWAC/domReport/ref_images/Zedlivevideo.avi'
      
        m_CAgthmParams = CAlgrthmParams(self.zed,sl,video_ocv,video_dpt,outputpth)

        m_CLanParams.FullRGBNumRowsInit = ArgAll.numL # initial lane settting of numbers for tracking
        m_CLanParams.NumoflaneInit = ArgAll.numL # initial lane settting of numbers for tracking
        m_CLanParams.FullDTHNumRowsInit= ArgAll.numL
        
        m_CAgthmParams.piplineSelet = ArgAll.pipL # Select different algorithm for detection lanes 1: for early tidy lane, 2: RGB highly adjoining lane, 3: Depth map on 2

        print ('Num of Lanes Setting = ', ArgAll.numL)
        #Algorithms encapsulatd here:
        # m_CLneDTrcks = CLneDetsTracks(m_CAgthmParams,m_CLanParams) 
        # m_CFRGBLneTrcks = CFullRGBLneDetsTracks(m_CAgthmParams,m_CLanParams)
        m_CFDptLneTrcks= CFullDpthLneDetsTracks(m_CAgthmParams,m_CLanParams)   


        m_zedcvHandle = zedcv(self.zed,sl,path) # instance of ZED stream
        # start_zed(zed, runtime, camera_pose, viewer, py_translation,depth_zed,image_zed,depth_image_zed,new_width,new_height,point_cloud,key)
        self.start_zed()      
    


    def stop_zed(self):
        global m_CamParams
        # zed_callback = threading.Thread(target=run, args=(m_CamParams.cam, m_CamParams.sl))
        self.stop_threads = True
        m_CamParams.cam.close()

    def start_zed(self):
        global m_CamParams
        # zed_callback = threading.Thread(target=run, args=(m_CamParams.cam, m_CamParams.sl))
        self.stop_threads = False
        zed_callback = threading.Thread(target=self.runproc,args =(lambda : self.stop_threads,)) # must be iterable so ','
        self.threads.append(zed_callback)
        zed_callback.start()


    # def run(cam, runtime, camera_pose, viewer, py_translation):
    def runproc(self,killed):       
        global m_CamParams
        # global m_CLneDTrcks
        global m_CAgthmParams
        global m_CFDptLneTrcks

        #open camera for streaming , temporarily for the test on video files collected
        # m_CLneDTrcks.CameraStreamfromfile()
        # m_CFRGBLneTrcks.CameraStreamfromfile()
        # m_CLneDTrcks.MainDeteTrack()#("/home/chfox/ARWAC/Essa-0/20190206_143625.mp4")
        m_CFDptLneTrcks.CameraStreamfromfile()
        print('zed camera is streaming, ...')

        while m_CamParams.key != 113 : #F2
            if killed() == True:
                print('Terminated...!')
                break
            err = m_CamParams.cam.grab(m_CamParams.runtime)
            if err == m_CamParams.sl.ERROR_CODE.SUCCESS :                
                                
                m_CamParams.cam.retrieve_measure(m_CamParams.depth_zed, m_CamParams.sl.MEASURE.DEPTH)  # application purpose
                # Load depth data into a numpy array
                m_CamParams.depth_ocv = m_CamParams.depth_zed.get_data()
                # Print the depth value at the center of the image
                print('center of image = ', m_CamParams.depth_ocv[int(len(m_CamParams.depth_ocv)/2)][int(len(m_CamParams.depth_ocv[0])/2)])

                m_CamParams.cam.retrieve_image(m_CamParams.image_zed, m_CamParams.sl.VIEW.LEFT) # Left image 
                # Use get_data() to get the numpy array === full
                m_CamParams.image_ocv = m_CamParams.image_zed.get_data()
                # report = np.hstack((image_ocv,depth_zed)) #stacking images side-by-side
                # # report = np.hstack((image_ocv,depth_zed)) #stacking images side-by-side
                # # Display the left image from the numpy array
                # # cv2.imshow("Image", image_ocv)
                # plt.figure('ZED Camera Live')
                # plt.title('depth_zed')        
                # plt.imshow(m_CamParams.depth_ocv,cmap='gray')
                # plt.show(block=False)
                # plt.pause(0.5)
                # plt.close()
                # Retrieve the left image, depth image in the half-resolution ----------- only for the display purpose
                # (m_CamParams.new_width, m_CamParams.new_height))
                m_CamParams.cam.retrieve_image(m_CamParams.image_zed, m_CamParams.sl.VIEW.LEFT, m_CamParams.sl.MEM.CPU, m_CamParams.image_size)
                # (m_CamParams.new_width, m_CamParams.new_height))
                m_CamParams.cam.retrieve_image(m_CamParams.depth_image_zed, m_CamParams.sl.VIEW.DEPTH, m_CamParams.sl.MEM.CPU, m_CamParams.image_size)

                # To recover data from sl.Mat to use it with opencv, use the get_data() method
                # It returns a numpy array that can be used as a matrix with opencv
                m_CamParams.image_ocv = m_CamParams.image_zed.get_data()
                m_CamParams.depth_image_ocv = m_CamParams.depth_image_zed.get_data()
                
                # if m_CAgthmParams.piplineSelet==1:
                # self.SimpfiedRGBPiplines(m_CamParams.image_ocv)
                #     pass    
                
                # elif m_CAgthmParams.piplineSelet==2:
                # self.RGBPiplines(m_CamParams.image_ocv)     
                    # pass          
                
                
                # m_CLneDptTrcks(m_CamParams.image_ocv,m_CamParams.depth_ocv)    
                # elif m_CAgthmParams.piplineSelet==3:
                self.DepthMapPiplines(m_CamParams.image_ocv,m_CamParams.depth_image_ocv)
                    # pass                 

                # switch_pipline(m_CAgthmParams.piplineSelet,m_CamParams.image_ocv)      
                # Retrieve the RGBA point cloud in half resolution           
                # m_CamParams.cam.retrieve_measure(m_CamParams.point_cloud, m_CamParams.sl.MEASURE.MEASURE_XYZRGBA, m_CamParams.sl.MEM.MEM_CPU, int(m_CamParams.new_width), int(m_CamParams.new_height)) 
               #################################
                # m_zedcvHandle.process_key_event(zed, key)
                #######################################################################
                # self.image_l = m_CLanParams.dispimg_l#m_CamParams.image_ocv
                # self.image_r = m_CLanParams.dispimg_r#m_CamParams.depth_image_ocv  

                
                # img = cv2.resize(m_CamParams.image_ocv,self.size_l, interpolation = cv2.INTER_AREA)
                img = cv2.resize(m_CAgthmParams.video_image,self.size_l, interpolation = cv2.INTER_AREA)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                self.image_l=Image.fromarray(img)
                self.image_l = ImageTk.PhotoImage(self.image_l)
                self.canvas_l.create_image(4, 4, anchor=NW, image=self.image_l) 

                img = cv2.resize(m_CamParams.depth_image_ocv,self.size_r, interpolation = cv2.INTER_AREA)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)                
                self.image_r =Image.fromarray(img)
                self.image_r = ImageTk.PhotoImage(self.image_r)
                self.canvas_r.create_image(4, 4, anchor=NW, image=self.image_r) 


                if self.image_l is None or self.image_r is None: 
                    return

                # img = cv2.resize(self.image_r,self.size_r, interpolation = cv2.INTER_AREA)
                # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # self.image_r = Image.fromarray(img)
                
                """
                img = cv2.imread("path/to/img.png")
                # You may need to convert the color.
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                im_pil = Image.fromarray(img)

                # For reversing the operation:
                im_np = np.asarray(im_pil)
                opencv_image=cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR) 
                """      
                
                # size_l = (self.width_l,self.height_l)
                # self.image_l = Image.fromarray(img)   
                # self.image_r.thumbnail(size_r, Image.ANTIALIAS)
                # self.image_r = self.image_r.resize((self.height_r, self.width_r), Image.ANTIALIAS) ## The (250, 250) is (height, width)
                # Label(self.right_frame, text="Input Stream").pack(side='top', padx=1, pady=1) 
                 
                
            else:
                # m_CamParams.sl.c_sleep_ms(1)
                key = cv2.waitKey(1)
        plt.close('all')
        cv2.destroyAllWindows()
        m_CamParams.cam.close()
        print("\nFINISH")

    # def switch_pipline(stcher, img):
    
    #     switcher = {
    #         0:  SimpfiedRGBPiplines(img),            
    #         1:  RGBPiplines(img),
    #         2:  DepthMapPiplines(img)      
        
    #     }
    # def SimpfiedRGBPiplines(self, img):   
    #     # plt.figure('ZED Camera Live')
    #     # plt.title('Visible and Depth')        
    #     # plt.imshow(img,cmap='gray')
    #     # plt.show(block=False)
    #     # plt.pause(0.25)
    #     # plt.close()
    #     m_CLneDTrcks.MainDeteTrack(img)

    # def RGBPiplines(self,img):   
    #     # plt.figure('ZED Camera Live')
    #     # plt.title('Visible and Depth')        
    #     # plt.imshow(img,cmap='gray')
    #     # plt.show(block=False)
    #     # plt.pause(0.25)
    #     # plt.close()
    #     m_CFRGBLneTrcks.MainDeteTrack(img)
    def DepthMapPiplines(self,img,dpth):   
        # plt.figure('ZED Camera Live')
        # plt.title('Visible and Depth')        
        # plt.imshow(img,cmap='gray')
        # plt.show(block=False)
        # plt.pause(0.25)
        # plt.close()
        m_CFDptLneTrcks.MainDeteTrack(img, dpth)


