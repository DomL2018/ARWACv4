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
# from tools.FullDepthLaneDeteTrack import CFullDpthLneDetsTracks
from argparse import ArgumentParser as ArgParse
import os

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

class GUIDepth(object):    
    def __init__(self, master, **kwargs):#__init__(self, title = "Lanes Tracking GUI", width = 800):
        """Initialise. Call this first before calling any other tkinter routines.
           title is the window title to use.
           width is the width of scale, bar etc. widgets.
        """
        self.root = master#Tk()
        self.width =800
        self.root.title("Lane Configuring Console")
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

    def toggle_fullscreen(self, event=None):
        self.state = not self.state  # Just toggling the boolean
        self.root.attributes("-fullscreen", self.state)
        return "break"

    def end_fullscreen(self, event=None):
        self.state = False
        self.root.attributes("-fullscreen", False)
        return "break"

    def initUI(self, event=None):

         # Create left and right frames
        self.l_frm_wdth = 960
        self.l_frm_hght = 640
        self.r_frm_wdth = 256
        self.r_frm_hght = 640

        self.left_frame = Frame(self.root, width=self.l_frm_wdth, height=self.l_frm_hght, bg='grey')
        self.left_frame.pack(side='left', fill='both', padx=8, pady=4, expand=True)

        self.right_frame = Frame(self.root, width=self.r_frm_wdth, height=self.r_frm_hght, bg='grey')
        self.right_frame.pack(side='right', fill='both', padx=8, pady=4, expand=True)
        ####################################################
        self.statusbar = Label(master = self.right_frame,text = "Ctrl.gui (tkinter {0})".format(TkVersion))
        self.statusbar.pack(padx = 1, pady = 1)
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
        self.image_l=cv2.imread("/home/dom/ARWAC/WeedRobLanCtrl/InitImages/20190516_142630.jpg")
        if self.image_l is None:
            w = Message(self.root, text="No image in InitGui", width=50)
            w.pack()

            
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
        Label(self.left_frame, text="Output processed").pack(side='top', padx=1, pady=1)  
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
        self.image_r=cv2.imread("/home/dom/ARWAC/WeedRobLanCtrl/InitImages/20190516_142759.jpg")
  

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
       
        return

    def set_status(self, text):
        """Display the new status text in the status bar. """
        self.statusbar["text"] = text
        return

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
        Button(master = self.tool_bar, text = label, command = self.get_threaded(callback)).pack(padx = 5, pady = 5)
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

        listbox.pack(padx = 5, pady = 5)

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
        value_label = Label(master = self.tool_bar,text = "0")
        value_label.pack(padx = 5, pady = 5)
        # TODO: Make callback threaded
        def callbackwrapper(value):
            # ttk.Scale calls back with a float string
            #
            value_label["text"] = int(float(value) * 10)
            callback(float(value))

        self.style.configure("TScale",orient = "horizontal",sliderlength = 10)
        # self.style.configure("TScale", padding=6, relief="flat",background="#ccc")
        self.scalwidth = int(self.scalwidth*0.5)
        # Scale(master = self.right_frame,length = self.scalwidth,command = callbackwrapper).pack(padx = 4, pady = 4)
        w = Scale(master= self.tool_bar, from_=10, to=-10,command = callbackwrapper)
        w.pack(padx = 10, pady = 5)        
        return

    def add_label(self, text):
        """Add a label.
        """
        Label(master = self.right_frame,text = text).pack(padx = 10, pady = 5)
        return

    def add_bar(self):
        """Add a bar for progress or meter display.
           Returns a function to be called with a float 0..1.
       """
        bar = ttk.Progressbar(master = self.filter_bar,
                                          orient = "horizontal",
                                          length = self.width,
                                          mode = "determinate",
                                          maximum = 100,
                                          value = 0)

        bar.pack(padx = 5, pady = 5)
        def callback(value):
            bar["value"] = int(value * 100)
            return
        return callback
    
    def add_canvas(self, width, height):
        #Add a canvas and return it. 
        canvas = Canvas(master = self.filter_bar, width = width, height = height)
        canvas.pack(padx = 5, pady = 5)
        return canvas
    
    def add_labelentry(self, text, width = 16, content = ""):
        """Create a labelled entry, and return the entry.
           width is the number of characters to show.
           content is a string to show in the entry.
           Call entry.get() to get its current content.
        """ 
        Label(master = self.filter_bar,text = "Data Input Below: ").pack(padx = 10, pady = 5)
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
        Label(master = self.filter_bar,text = scalelabel).pack(padx = 10, pady = 5)

        rating_scale_frame = Frame(master = self.filter_bar)
        Label(master = rating_scale_frame,text = worst_label).pack(side = LEFT, padx = 5)
        scale_value = IntVar()
        radiobuttons = []        
        for i in range(count):            
            r = Radiobutton(master = rating_scale_frame,
                                    text = "",
                                    variable = scale_value,
                                    value = i)

            r.pack(side = LEFT, padx = 5)
            radiobuttons.append(r)

        # Set center button active
        # Note: .select() is not present in ttk.Radiobuttin
        #
        radiobuttons[int(len(radiobuttons) / 2)].invoke()        
        Label(master = rating_scale_frame,
                      text = best_label).pack(side = LEFT, padx = 5)
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
        init.camera_resolution = sl.RESOLUTION.RESOLUTION_HD1080
        init.depth_mode = sl.DEPTH_MODE.DEPTH_MODE_ULTRA#.DEPTH_MODE_PERFORMANCE
        init.coordinate_units = sl.UNIT.UNIT_METER
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
    
        m_CLanParams = CLanesParams()
        m_CamParams = CCamsParams(self.zed, sl)


        # inputpth ="/home/chfox/ARWAC/Essa-0/20190206_143625.mp4"  
        video_ocv = "/home/chfox/ARWAC/zed_avi_l/VGA_SN12925_13-29-42_l.avi" 
        video_dpt = "/home/chfox/ARWAC/zed_avi_l/VGA_SN12925_13-29-42_d.avi"     
        outputpth ='/home/chfox/Documents/ARWAC/domReport/ref_images/Zedlivevideo.avi'
        # video_ocv ="/home/chfox/ARWAC/Essa-0/20190206_143625.mp4"  
        # video_dpt = ''        
        m_CAgthmParams = CAlgrthmParams(self.zed,sl,video_ocv,video_dpt,outputpth)

        # m_CLanParams.FullRGBNumRowsInit = ArgAll.numL # initial lane settting of numbers for tracking
        # m_CLanParams.NumoflaneInit = ArgAll.numL # initial lane settting of numbers for tracking
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
    


    def start_zed(self):
        global m_CamParams
        # zed_callback = threading.Thread(target=run, args=(m_CamParams.cam, m_CamParams.sl))
        self.stop_threads = False
        zed_callback = threading.Thread(target=self.runproc,args =(lambda : self.stop_threads,)) # must be iterable so ','
        zed_callback.start()


    # def run(cam, runtime, camera_pose, viewer, py_translation):
    def runproc(self,killed):       
        global m_CamParams
        global m_CFDptLneTrcks
        global m_CAgthmParams

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

                
                                
                m_CamParams.cam.retrieve_measure(m_CamParams.depth_zed, m_CamParams.sl.MEASURE.MEASURE_DEPTH) # application purpose
                # Load depth data into a numpy array
                m_CamParams.depth_ocv = m_CamParams.depth_zed.get_data()
                # Print the depth value at the center of the image
                print('center of image = ', m_CamParams.depth_ocv[int(len(m_CamParams.depth_ocv)/2)][int(len(m_CamParams.depth_ocv[0])/2)])

                m_CamParams.cam.retrieve_image(m_CamParams.image_zed, m_CamParams.sl.VIEW.VIEW_LEFT) # Left image 
                # Use get_data() to get the numpy array === full
                m_CamParams.image_ocv = m_CamParams.image_zed.get_data()
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
                m_CamParams.cam.retrieve_image(m_CamParams.image_zed, m_CamParams.sl.VIEW.VIEW_LEFT, m_CamParams.sl.MEM.MEM_CPU, int(m_CamParams.new_width), int(m_CamParams.new_height))
                m_CamParams.cam.retrieve_image(m_CamParams.depth_image_zed, m_CamParams.sl.VIEW.VIEW_DEPTH, m_CamParams.sl.MEM.MEM_CPU, int(m_CamParams.new_width), int(m_CamParams.new_height))
                        
                # To recover data from sl.Mat to use it with opencv, use the get_data() method
                # It returns a numpy array that can be used as a matrix with opencv
                m_CamParams.image_ocv = m_CamParams.image_zed.get_data()
                m_CamParams.depth_image_ocv = m_CamParams.depth_image_zed.get_data()

                # report = np.hstack((image_ocv,depth_ocv)) #stacking images side-by-side
                # report = np.hstack((m_CamParams.image_ocv,m_CamParams.depth_image_ocv)) #stacking images side-by-side
                # # cv2.imwrite('/home/chfox/Documents/ARWAC/FarmLanedetection/ref_images/Fig18EdgsDepth.jpg',regionofintrest)
                # # cv2.imwrite('/home/chfox/Documents/ARWAC/FarmLanedetection/ref_images/Fig18HoughLineDepth.jpg',masked)
                # # cv2.imwrite('/home/chfox/Documents/ARWAC/FarmLanedetection/ref_images/Fig18EdgeandHhLinesDepth.jpg',report)
                # plt.figure('ZED Camera Live')
                # plt.title('Visible and Depth')        
                # plt.imshow(report,cmap='gray')
                # plt.show(block=False)
                # plt.pause(0.5)
                # plt.close()
                
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
                # self.image_l=Image.fromarray(m_CamParams.image_ocv)
                # self.image_r =Image.fromarray(m_CamParams.depth_image_ocv)
                if self.image_l is None or self.image_r is None: 
                    return

                self.image_l = m_CLanParams.dispimg_l#m_CamParams.image_ocv
                self.image_r = m_CLanParams.dispimg_r#m_CamParams.depth_image_ocv
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
                img = cv2.resize(self.image_l,self.size_l, interpolation = cv2.INTER_AREA)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                self.image_l = Image.fromarray(img)

                self.image_l = ImageTk.PhotoImage(self.image_l)
                self.canvas_l.create_image(4, 4, anchor=NW, image=self.image_l) 
               
             
                img = cv2.resize(self.image_r,self.size_r, interpolation = cv2.INTER_AREA)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                self.image_r = Image.fromarray(img)
                # self.image_r.thumbnail(size_r, Image.ANTIALIAS)

                # self.image_r = self.image_r.resize((self.height_r, self.width_r), Image.ANTIALIAS) ## The (250, 250) is (height, width)
                # Label(self.right_frame, text="Input Stream").pack(side='top', padx=1, pady=1) 
                self.image_r = ImageTk.PhotoImage(self.image_r)
                self.canvas_r.create_image(4, 4, anchor=NW, image=self.image_r)    

                
                
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
    #     m_CLneDTrcks.MainDeteTrack(img)
    # def RGBPiplines(self,img):   
    #     m_CFRGBLneTrcks.MainDeteTrack(img)
    def DepthMapPiplines(self,ocv,dpth):   
        m_CFDptLneTrcks.MainDeteTrack(ocv,dpth)


