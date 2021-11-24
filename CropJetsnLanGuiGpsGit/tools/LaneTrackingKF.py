from __future__ import print_function
print(__doc__)
import sys
PY3 = sys.version_info[0] == 3
import cv2
import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
# from sklearn.datasets.samples_generator import make_blobs
import matplotlib.pyplot as plt
# import pylab
import math
import statistics
# import argparse

import os, glob
# from moviepy.editor import VideoFileClip
# built-in modules
import itertools as it
from itertools import count
font = cv2.FONT_HERSHEY_SIMPLEX
class grasstrack():
    """Pedestrian class

    each pedestrian is composed of a ROI, an ID and a Kalman filter
    so we create a Pedestrian class to hold the object state
    """
    _ids = count(1)  
    # link Kalman with Object（self.kalman）
    def __init__(self, id, frame, track_pt,track_window,lines_mid_vec,bandwith,lanegap):
        """init the pedestrian object with track window coordinates"""
        # set up the roi
        self.id = id
        #counting instances of a grasstrack
        self.numofids  = next(self._ids)
        print('total of ids = {0}, lane ID = {1}'.format(self.numofids,self.id))
        x,y,w,h = track_window
        self.track_window = track_window
        self.lines_mid_vec = lines_mid_vec
        self.bandwith=bandwith
        self.lanegap = lanegap
        self.roi = frame[y-h:y, x-w:x]#cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2HSV)
        # roi_hist = cv2.calcHist([self.roi], [0], None, [16], [0, 180])
        # self.roi_hist = cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

        # set up the kalman
        self.kalman = cv2.KalmanFilter(4,2)
        self.kalman.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]],np.float32)
        self.kalman.transitionMatrix = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]],np.float32)
        self.kalman.processNoiseCov = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],np.float32) * 0.1  #0.03
        self.measurement = np.array((2,1), np.float32) 
        self.prediction = np.zeros((2,1), np.float32)
        self.term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )
        self.track_pt=track_pt
        self.center = track_pt
        self.prediction = track_pt
        self.pre_updated_id = id
        # self.update(frame,track_pt,track_window)
        
    def __del__(self):
        print ("lane %d destroyed" %self.id)

    def update(self, frame,track_pt,track_window):
        # print("updating id = %d" %self.id)
        self.track_pt=track_pt
        self.track_window=track_window
        # print "updating %d " % self.id
        algorithm=0
        # if args.get("algorithm") == "c":
        if algorithm == 1:
        #   ret, self.track_window = cv2.CamShift(back_project, self.track_window, self.term_crit)
        #   pts = cv2.boxPoints(ret)
        #   ret=self.track_window
        #   pts = cv2.boxPoints(ret)
        #   pts = np.int0(pts)
        #   self.center = center(pts)
        #   cv2.polylines(frame,[pts],True, 255,1)
            x,y,w,h = self.track_window
            x=x-w
            y=y-h
            img2 = cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,125),2)
        
        # lane_coords=[]
        # datacx=[]
        # datacy=[]
        # for point in self.lines_mid_vec: 
        #     if abs(self.id-point[0][0])<self.bandwith:
        #             cx=point[0][0]
        #             cy=point[0][1]
        #             datacx.append(cx)
        #             datacy.append(cy)
        #             lane_coords.append((cx, cy)) 

        # self.cx_median= statistics.median(datacx)
        # # print("sefl.cx_median",self.cx_median)
        # self.cy_median= statistics.median(datacy)
        # print("sefl.cy_median",self.cy_median)
        # self.center=np.median(lane_coords)      #update the center points , which will be used as the next update/correction of Kalman filter           
        # print('median center',self.center)
        # self.center=np.array([np.float32(self.cx_median), np.float32(self.cy_median)], np.float32)
        
        self.center=np.array([np.float32(self.track_pt[0]), np.float32(self.track_pt[1])], np.float32)
        # print("stracking state: x= %4.2f, and  y= %4.2f"%(self.center[0],self.center[1]))

        self.center=self.kalman.correct(self.center)
        width = int(frame.shape[1])
        height = int(frame.shape[0])
        # print("state correction: x=%4.2f, and y= %4.2f "%(self.center[0], self.center[1]))
        tmx=self.center[0]
        tmy=self.center[1]

        if self.center[0]>width:
            tmx=width-4
        if self.center[1]>height:
            tmy=height-4

        cv2.circle(frame, (int(tmx), int(tmy)), 4, (0, 0,255), -1) 
        frame = cv2.line(frame,(int(tmx),height-1),(int(tmx),0),(255,255,0),4)

        # cv2.circle(frame, (int(self.center[0]), int(self.center[1])), 4, (0, 0,255), -1) 
        # frame = cv2.line(frame,(int(self.center[0]),height-1),(int(self.center[0]),0),(255,255,0),4)
        self.prediction = self.kalman.predict()      
        # print("state prediction:x=%4.2f, and y= %4.2f "%(self.prediction[0],self.prediction[1]))  
        # self.id=int(self.prediction[0])
        # cv2.circle(frame, (int(prediction[0]), int(prediction[1])), 4, (255, 0, 0), -1)     
        return frame    