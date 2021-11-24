#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import threading
if sys.version_info[0] == 2:  # Just checking your Python version to import Tkinter properly.
    from Tkinter import *
else:
    from tkinter import *
    from tkinter import ttk

from tools.FullDPTCtrlGui import GUI
from PIL import ImageTk, Image

import numpy as np
import pyzed.sl as sl
import cv2
from tools.zedcvstream import zedcv # import zed processing methods
from tools.CSysParams import CLanesParams,CAlgrthmParams,CCamsParams
import matplotlib.pyplot as plt
import threading

# from tools.LaneDeteTrack import CLneDetsTracks
# from tools.FullRGBLaneDeteTrack import CFullRGBLneDetsTracks   
from tools.FullDptLaneDeteTrack import CFullDpthLneDetsTracks   
from argparse import ArgumentParser as ArgParse
import os

# path = "/home/dom/prototype/ZED/"
# m_CLanParams = object()
# m_CAgthmParams = object()
# m_CamParams  = object()
# # m_CLneDTrcks = object()
# m_CFRGBLneTrcks = object()
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
        self.gg = GUI(self.root)

    def toggle_fullscreen(self, event=None):
        self.state = not self.state  # Just toggling the boolean
        self.root.attributes("-fullscreen", self.state)
        return "break"

    def end_fullscreen(self, event=None):
        self.state = False
        self.root.attributes("-fullscreen", False)
        return "break"

    def monicols(self,AllArgs):       
        
        
        def listboxcallback(text):
            self.g.set_status("Method Select: '{0}'".format(text))
        self.gg.add_listbox(["Method 1", "Method 2", "Method 3"], listboxcallback)
        # g.add_listbox(["A", "B", "C"], listboxcallback) 
        
        def scalecallback(text):
            self.gg.set_status("scale value: '{0:.2}'".format(text))
        self.gg.add_scale("Scale adjusting!", scalecallback)
        # self.gg.add_label("Following Setting ...")
        """
        entry = self.gg.add_labelentry("Type:", content = "Data  is ready.")
        def get_entry():
            self.gg.set_status(entry.get())
        self.gg.add_button(" Info to entry", get_entry)
        

        rating_scale = self.gg.add_rating_scale("Image Chosen:", 2, "RGB", "Depth")
        def get_rating():
            self.gg.set_status(rating_scale.get())
        self.gg.add_button("Select Image Type", get_rating)
       """
        
        self.gg.add_statusbar()
        
        # We need a mutable object to access it from the callback, so we use a list
        bar_callback = self.gg.add_ImgOffsetBar()

        offset_bar_value = [0]
        def increase_offset_bar():
            offset_bar_value[0] = 2# not for accumulating , only step up a unit #+2 n#0.1 #
            self.gg.m_OffsetCtrlBar=True # increae
            if offset_bar_value[0] > 200:
                offset_bar_value[0] = 0
            bar_callback(offset_bar_value[0])
            
        self.gg.add_button("Image Offset ++", increase_offset_bar)

        def decrease_offset_bar():
            offset_bar_value[0] = 2# not for accumulating , only step up a unit #-2 n#0.1 #
            self.gg.m_OffsetCtrlBar=False # decrease
            if offset_bar_value[0] < 1:
                offset_bar_value[0] = 0
            bar_callback(offset_bar_value[0])            
        self.gg.add_button("Image Offset --", decrease_offset_bar)

        # set lane number setting default 4 currently        
        self.gg.add_LaneSetstatusbar()
        lanesetingbar_callback = self.gg.add_LaneSettingBar()

        laneSet_bar_value = [0]
        def increase_laneSetting_bar():
            laneSet_bar_value[0] = 1# not for accumulating , only step up a unit #+2 n#0.1 #
            self.gg.m_LaneSetCtrlBar=True # increae
            if laneSet_bar_value[0] > 20:
                laneSet_bar_value[0] = 0
            lanesetingbar_callback(laneSet_bar_value[0])
            
        self.gg.add_button("Num of Lane Setting ++", increase_laneSetting_bar)

        def decrease_laneSetting_bar():
            laneSet_bar_value[0] = 1# not for accumulating , only step up a unit #-2 n#0.1 #            
            self.gg.m_LaneSetCtrlBar=False # decrease
            if laneSet_bar_value[0] < 1:
                laneSet_bar_value[0] = 0
            lanesetingbar_callback(laneSet_bar_value[0])

        self.gg.add_button("Num of Lane Setting --", decrease_laneSetting_bar)

        # set lane starting points default 25 currently        
        self.gg.add_LaneOriPtstatusbar()
        laneOriPts_callback = self.gg.add_LaneOriPtsBar()

        laneOriPtsSet_bar_value = [0]
        def inc_laneOriPtsSetting_bar():
            laneOriPtsSet_bar_value[0] = 1# not for accumulating , only step up a unit #+2 n#0.1 #
            self.gg.m_LaneOriSetCtrlBar=True # increae
            if laneOriPtsSet_bar_value[0] > 200:
                laneOriPtsSet_bar_value[0] = 0
            laneOriPts_callback(laneOriPtsSet_bar_value[0])            
        self.gg.add_button("Lane Start Setting ++", inc_laneOriPtsSetting_bar)

        def dec_laneOriPtsSetting_bar():
            laneOriPtsSet_bar_value[0] = 1# not for accumulating , only step up a unit #-2 n#0.1 #            
            self.gg.m_LaneOriSetCtrlBar=False # decrease
            if laneOriPtsSet_bar_value[0] < 1:
                laneOriPtsSet_bar_value[0] = 0
            laneOriPts_callback(laneOriPtsSet_bar_value[0])
        self.gg.add_button("Lane Start Setting --", dec_laneOriPtsSetting_bar)


        # set lane bandwith setting default 75 currently      

        self.gg.add_LaneBandwithstatusbar()
        laneBandwith_callback = self.gg.add_LaneBandwithBar()

        laneBandwithSet_bar_value = [0]
        def inc_laneBandwithSetting_bar():
            laneBandwithSet_bar_value[0] = 1# not for accumulating , only step up a unit #+2 n#0.1 #
            self.gg.m_LaneBandwithCtrlBar=True # increae
            if laneBandwithSet_bar_value[0] > 200:
                laneBandwithSet_bar_value[0] = 0
            laneBandwith_callback(laneBandwithSet_bar_value[0])            
        self.gg.add_button("Lane Bandwith Setting ++", inc_laneBandwithSetting_bar)

        def dec_laneBandwithSetting_bar():
            laneBandwithSet_bar_value[0] = 1# not for accumulating , only step up a unit #-2 n#0.1 #            
            self.gg.m_LaneBandwithCtrlBar=False # decrease
            if laneBandwithSet_bar_value[0] < 1:
                laneBandwithSet_bar_value[0] = 0
            laneBandwith_callback(laneBandwithSet_bar_value[0])
        self.gg.add_button("Lane Bandwith Setting --", dec_laneBandwithSetting_bar)


        # set minimum line lengh for hough transform used default 30,   

        self.gg.add_minLineLengthstatusbar()
        minLineLength_callback = self.gg.add_minLineLengthBar()
        minLineLength_bar_value = [0]
        
        def inc_minLineLengthSetting_bar():
            minLineLength_bar_value[0] = 1# not for accumulating , only step up a unit #+2 n#0.1 #
            self.gg.m_minLineLengthCtrlBar=True # increae
            if minLineLength_bar_value[0] > 200:
                minLineLength_bar_value[0] = 0
            minLineLength_callback(minLineLength_bar_value[0])            
        self.gg.add_button("minLineLength Setting ++", inc_minLineLengthSetting_bar)

        def dec_minLineLengthSetting_bar():
            minLineLength_bar_value[0] = 1# not for accumulating , only step up a unit #-2 n#0.1 #            
            self.gg.m_minLineLengthCtrlBar=False # decrease
            if minLineLength_bar_value[0] < 1:
                minLineLength_bar_value[0] = 0
            minLineLength_callback(minLineLength_bar_value[0])
        self.gg.add_button("minLineLength Setting  --", dec_minLineLengthSetting_bar)


        # set maxGap line lengh for hough transform used default 4, 

        self.gg.add_maxGapstatusbar()
        maxGap_callback = self.gg.add_maxGapBar()
        maxGap_bar_value = [0]
        
        def inc_maxGapSetting_bar():
            maxGap_bar_value[0] = 1# not for accumulating , only step up a unit #+2 n#0.1 #
            self.gg.m_maxGapCtrlBar=True # increae
            if maxGap_bar_value[0] > 200:
                maxGap_bar_value[0] = 0
            maxGap_callback(maxGap_bar_value[0])            
        self.gg.add_button("maxGap Setting ++", inc_maxGapSetting_bar)

        def dec_maxGapSetting_bar():
            maxGap_bar_value[0] = 1# not for accumulating , only step up a unit #-2 n#0.1 #            
            self.gg.m_maxGapCtrlBar=False # decrease
            if maxGap_bar_value[0] < 1:
                maxGap_bar_value[0] = 1
            maxGap_callback(maxGap_bar_value[0])
        self.gg.add_button("maxGap Setting  --", dec_maxGapSetting_bar)

    
        # set green color low base in HSV sapce , default 42, 

        self.gg.add_greenLowclorstatusbar()
        greenLowclor_callback = self.gg.add_greenLowclorBar()
        greenLowclor_bar_value = [0]
        
        def inc_greenLowclorSetting_bar():
            greenLowclor_bar_value[0] = 1# not for accumulating , only step up a unit #+2 n#0.1 #
            self.gg.m_greenLowclorCtrlBar=True # increae
            if greenLowclor_bar_value[0] > 180:
                greenLowclor_bar_value[0] = 1
            greenLowclor_callback(greenLowclor_bar_value[0])            
        self.gg.add_button("Lowclor Setting ++", inc_greenLowclorSetting_bar)

        def dec_greenLowclorSetting_bar():
            greenLowclor_bar_value[0] = 1# not for accumulating , only step up a unit #-2 n#0.1 #            
            self.gg.m_greenLowclorCtrlBar=False # decrease
            if greenLowclor_bar_value[0] < 25:
                greenLowclor_bar_value[0] = 1
            greenLowclor_callback(greenLowclor_bar_value[0])
        self.gg.add_button("Lowclor Setting  --", dec_greenLowclorSetting_bar)


        # set green color High base in HSV sapce , default 106, 

        self.gg.add_greenHigclorstatusbar()
        greenHigclor_callback = self.gg.add_greenHigclorBar()
        greenHigclor_bar_value = [0]
        
        def inc_greenHigclorSetting_bar():
            greenHigclor_bar_value[0] = 1# not for accumulating , only step up a unit #+2 n#0.1 #
            self.gg.m_greenHigclorCtrlBar=True # increae
            if greenHigclor_bar_value[0] > 180:
                greenHigclor_bar_value[0] = 1
            greenHigclor_callback(greenHigclor_bar_value[0])            
        self.gg.add_button("Higclor Setting ++", inc_greenHigclorSetting_bar)

        def dec_greenHigclorSetting_bar():
            greenHigclor_bar_value[0] = 1# not for accumulating , only step up a unit #-2 n#0.1 #            
            self.gg.m_greenHigclorCtrlBar=False # decrease
            if greenHigclor_bar_value[0] < 2:
                greenHigclor_bar_value[0] = 1
            greenHigclor_callback(greenHigclor_bar_value[0])
        self.gg.add_button("Higclor Setting  --", dec_greenHigclorSetting_bar)

        # set Low saturation value HSV sapce , default 25, 

        self.gg.add_greenLowSaturstatusbar()
        greenLowSatur_callback = self.gg.add_greenLowSaturBar()
        greenLowSatur_bar_value = [0]
        
        def inc_greenLowSaturSetting_bar():
            greenLowSatur_bar_value[0] = 1# not for accumulating , only step up a unit #+2 n#0.1 #
            self.gg.m_greenLowSaturCtrlBar=True # increae
            if greenLowSatur_bar_value[0] > 225:
                greenLowSatur_bar_value[0] = 1
            greenLowSatur_callback(greenLowSatur_bar_value[0])            
        self.gg.add_button("LowSatur Setting ++", inc_greenLowSaturSetting_bar)

        def dec_greenLowSaturSetting_bar():
            greenLowSatur_bar_value[0] = 1# not for accumulating , only step up a unit #-2 n#0.1 #            
            self.gg.m_greenLowSaturCtrlBar=False # decrease
            if greenLowSatur_bar_value[0] < 2:
                greenLowSatur_bar_value[0] = 1
            greenLowSatur_callback(greenLowSatur_bar_value[0])
        self.gg.add_button("LowSatur Setting  --", dec_greenLowSaturSetting_bar)

        # set Low value (brightness)  in HSV sapce , default 25, 

        self.gg.add_greenLowValstatusbar()
        greenLowVal_callback = self.gg.add_greenLowValBar()
        greenLowVal_bar_value = [0]
        
        def inc_greenLowValSetting_bar():
            greenLowVal_bar_value[0] = 1# not for accumulating , only step up a unit #+2 n#0.1 #
            self.gg.m_greenLowValCtrlBar=True # increae
            if greenLowVal_bar_value[0] > 180:
                greenLowVal_bar_value[0] = 1
            greenLowVal_callback(greenLowVal_bar_value[0])            
        self.gg.add_button("LowVal Setting ++", inc_greenLowValSetting_bar)

        def dec_greenLowValSetting_bar():
            greenLowVal_bar_value[0] = 1# not for accumulating , only step up a unit #-2 n#0.1 #            
            self.gg.m_greenLowValCtrlBar=False # decrease
            if greenLowVal_bar_value[0] < 2:
                greenLowVal_bar_value[0] = 1
            greenLowVal_callback(greenLowVal_bar_value[0])
        self.gg.add_button("LowVal Setting  --", dec_greenLowValSetting_bar)
       
        
        #######################################################


        # set croping row0 setting : starting cropping point for row from origin raw image, 
        self.gg.add_FullRGBRow0Statusbar()
        FullRGBRow0_callback = self.gg.add_FullRGBRow0Bar()
        FullRGBRow0_bar_value = [0]
        
        def inc_FullRGBRow0Setting_bar():
            FullRGBRow0_bar_value[0] = 1# not for accumulating , only step up a unit #+2 n#0.1 #
            self.gg.m_FullRGBRow0CtrlBar=True # increae
            if FullRGBRow0_bar_value[0] > 180:
                FullRGBRow0_bar_value[0] = 1
            FullRGBRow0_callback(FullRGBRow0_bar_value[0])            
        self.gg.add_button("FullRGBRow0 Setting ++", inc_FullRGBRow0Setting_bar)

        def dec_FullRGBRow0Setting_bar():
            FullRGBRow0_bar_value[0] = 1# not for accumulating , only step up a unit #-2 n#0.1 #            
            self.gg.m_FullRGBRow0CtrlBar=False # decrease
            if FullRGBRow0_bar_value[0] < 2:
                FullRGBRow0_bar_value[0] = 1
            FullRGBRow0_callback(FullRGBRow0_bar_value[0])
        self.gg.add_button("FullRGBRow0 Setting  --", dec_FullRGBRow0Setting_bar)
       

        #######################################################

         # set croping h0 setting : cropped imag height, 

        self.gg.add_FullRGBH0Statusbar()
        FullRGBH0_callback = self.gg.add_FullRGBH0Bar()
        FullRGBH0_bar_value = [0]
        
        def inc_FullRGBH0Setting_bar():
            FullRGBH0_bar_value[0] = 1# not for accumulating , only step up a unit #+2 n#0.1 #
            self.gg.m_FullRGBH0CtrlBar=True # increae
            if FullRGBH0_bar_value[0] > 180:
                FullRGBH0_bar_value[0] = 1
            FullRGBH0_callback(FullRGBH0_bar_value[0])            
        self.gg.add_button("FullRGBH0 Setting ++", inc_FullRGBH0Setting_bar)

        def dec_FullRGBH0Setting_bar():
            FullRGBH0_bar_value[0] = 1# not for accumulating , only step up a unit #-2 n#0.1 #            
            self.gg.m_FullRGBH0CtrlBar=False # decrease
            if FullRGBH0_bar_value[0] < 2:
                FullRGBH0_bar_value[0] = 1
            FullRGBH0_callback(FullRGBRow0_bar_value[0])
        self.gg.add_button("FullRGBH0 Setting  --", dec_FullRGBH0Setting_bar)       

        #######################################################        

         # set croping col0 setting : cropped imag starting col point, 

        self.gg.add_FullRGBCol0Statusbar()
        FullRGBCol0_callback = self.gg.add_FullRGBCol0Bar()
        FullRGBCol0_bar_value = [0]
        
        def inc_FullRGBCol0Setting_bar():
            FullRGBCol0_bar_value[0] = 1# not for accumulating , only step up a unit #+2 n#0.1 #
            self.gg.m_FullRGBCol0CtrlBar=True # increae
            if FullRGBCol0_bar_value[0] > 180:
                FullRGBCol0_bar_value[0] = 1
            FullRGBCol0_callback(FullRGBCol0_bar_value[0])            
        self.gg.add_button("FullRGBCol0 Setting ++", inc_FullRGBCol0Setting_bar)

        def dec_FullRGBCol0Setting_bar():
            FullRGBCol0_bar_value[0] = 1# not for accumulating , only step up a unit #-2 n#0.1 #            
            self.gg.m_FullRGBCol0CtrlBar=False # decrease
            if FullRGBCol0_bar_value[0] < 2:
                FullRGBCol0_bar_value[0] = 1
            FullRGBCol0_callback(FullRGBCol0_bar_value[0])
        self.gg.add_button("FullRGBCol0 Setting  --", dec_FullRGBCol0Setting_bar)       

        #######################################################
        
        # set croping col0 setting : cropped imag width
        self.gg.add_FullRGBW0Statusbar()
        FullRGBW0_callback = self.gg.add_FullRGBW0Bar()
        FullRGBW0_bar_value = [0]
        
        def inc_FullRGBW0Setting_bar():
            FullRGBW0_bar_value[0] = 1# not for accumulating , only step up a unit #+2 n#0.1 #
            self.gg.m_FullRGBW0CtrlBar=True # increae
            if FullRGBW0_bar_value[0] > 180:
                FullRGBW0_bar_value[0] = 1
            FullRGBW0_callback(FullRGBW0_bar_value[0])            
        self.gg.add_button("FullRGBW0 Setting ++", inc_FullRGBW0Setting_bar)

        def dec_FullRGBW0Setting_bar():
            FullRGBW0_bar_value[0] = 1# not for accumulating , only step up a unit #-2 n#0.1 #            
            self.gg.m_FullRGBW0CtrlBar=False # decrease
            if FullRGBW0_bar_value[0] < 2:
                FullRGBW0_bar_value[0] = 1
            FullRGBW0_callback(FullRGBW0_bar_value[0])
        self.gg.add_button("FullRGBW0 Setting --", dec_FullRGBW0Setting_bar)       

        #######################################################
        def InitBTCallback():
            self.gg.set_status("Initializing...")
            self.gg.startproces(AllArgs)
            # arwacmain(AllArgs)
        self.gg.add_button2("Initialize!", InitBTCallback)  

        def startbuttoncallback():
            self.gg.set_status("Starting...")
            # self.gg.startproces(AllArgs)
            self.gg.SetStart_Ctrl()
            # arwacmain(AllArgs)
        self.gg.add_button2("Start !", startbuttoncallback)  

        def pausebuttoncallback():
            self.gg.set_status("Paused !")
        self.gg.add_button("Pause !", pausebuttoncallback)
        self.gg.add_button("Quit", lambda: self.gg.destroy())

        def run_when_started():
            self.gg.set_status("Ready to be executed")
            # arwacmain(AllArgs)
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
