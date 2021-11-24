import sys
import numpy as np
from numpy import inf
# import pyzed.sl as sl
import cv2


# help_string = "[s] Save side by side image [d] Save Depth, [n] Change Depth format, [p] Save Point Cloud, [m] Change Point Cloud format, [q] Quit"
# prefix_point_cloud = "Cloud_"
# prefix_depth = "Depth_"
# #path = "./"
# path = "/home/chfox/prototype/ZED/"
# count_save = 0
# mode_point_cloud = 0
# mode_depth = 0
# point_cloud_format = sl.POINT_CLOUD_FORMAT.POINT_CLOUD_FORMAT_XYZ_ASCII
# depth_format = sl.DEPTH_FORMAT.DEPTH_FORMAT_PNG

class CVideoSrcSetting(object):
    def __init__(self):
        self.DataSets={}#FullRRGNumRowsInit=4 #total number of rows covered within ROI
        self.FullDTHNumRowsInit=4 #total number of rows covered within ROI
        self.recording_counter=0 
        self.deltalane = '' # cut lanes for initilaization
        
        self.SIMPNumoflaneInit =6 #total number of rows covered within ROI

        self.SIMPbandwith=75  # setting on 'tidy ' image
        self.SIMPRGBrow0 = 20
        self.SIMPRGBh0=1880
        self.SIMPRGBcol0 = 50
        self.SIMPRGBw0=1000
        self.SIMPLaneOriPoint=25#origin point for tracking 
    def Data_28072020_1st(self):

        self.DataSets['pitch']= str(60)
        self.DataSets['path']="/media/dom/Elements/data/TX2_Risehm/28072020/1st/goVL720q.avi"
        self.DataSets['row0']=350
        self.DataSets['col0']=175
        self.DataSets['w0']=1025
        self.DataSets['lanebandwidth']=80

        self.DataSets['NumofLanes']=6
        self.DataSets['laneoffset']=155
        self.DataSets['LaneOriPoint']=50
        self.DataSets['frm2strart']=980

        self.DataSets['frame_width']=1025
        self.DataSets['frame_height']=370
        self.DataSets['low_green']=np.array([30, 0, 0])
        self.DataSets['high_green']=np.array([100, 255, 255])
        
        return self.DataSets
    def Data_28072020_2nd(self):

        self.DataSets['pitch']= str(80)
        self.DataSets['path']="/media/dom/Elements/data/TX2_Risehm/28072020/2nd/goVL720q.avi"    
        self.DataSets['row0']=0
        self.DataSets['col0']=100
        self.DataSets['w0']=1200
        self.DataSets['lanebandwidth']=80

        self.DataSets['h0']=500

        self.DataSets['NumofLanes']=6
        self.DataSets['laneoffset']=40
        self.DataSets['LaneOriPoint']=40
        self.DataSets['frm2strart']=400

        self.DataSets['frame_width']=1200
        self.DataSets['frame_height']=500
        self.DataSets['low_green']=np.array([30, 0, 0])
        self.DataSets['high_green']=np.array([100, 255, 255])
        
        return self.DataSets

    def Data_28072020_3rd(self):

        self.DataSets['pitch']= str(45)
        self.DataSets['path']="/media/dom/Elements/data/TX2_Risehm/28072020/3rd/goVL720q.avi"    
        self.DataSets['row0']=500
        self.DataSets['col0']=250
        self.DataSets['w0']=900
        self.DataSets['lanebandwidth']=80

        self.DataSets['NumofLanes']=6
        self.DataSets['laneoffset']=104
        self.DataSets['LaneOriPoint']=40
        self.DataSets['frm2strart']=175

        self.DataSets['frame_width']=900
        self.DataSets['frame_height']=220
        self.DataSets['low_green']=np.array([30, 0, 0])
        self.DataSets['high_green']=np.array([100, 255, 255])
        
        return self.DataSets
    def Data_28072020_4th(self):

        self.DataSets['pitch']= str(30)
        self.DataSets['path']="/media/dom/Elements/data/TX2_Risehm/28072020/4th/goVL720p.avi"
        self.DataSets['row0']=400
        self.DataSets['col0']=120
        self.DataSets['w0']=1100
        self.DataSets['lanebandwidth']=80

        self.DataSets['NumofLanes']=6
        self.DataSets['laneoffset']=80
        self.DataSets['LaneOriPoint']=40
        self.DataSets['frm2strart']=450

        self.DataSets['frame_width']=110
        self.DataSets['frame_height']=320

        self.DataSets['low_green']=np.array([30, 0, 0])
        self.DataSets['high_green']=np.array([100, 255, 255])
        
        return self.DataSets

    #################################################################

    def Data_31072020_10801st(self):

        self.DataSets['pitch']= str(30)
        self.DataSets['path']="/media/dom/Elements/data/TX2_Risehm/data_310720_1st/1080p1st/goVL1080p.avi"
        self.DataSets['row0']=20
        self.DataSets['col0']=175
        self.DataSets['w0']=1500
        self.DataSets['lanebandwidth']=80

        self.DataSets['NumofLanes']=6
        self.DataSets['laneoffset']=50
        self.DataSets['LaneOriPoint']=30
        self.DataSets['frm2strart']=180

        self.DataSets['frame_width']=self.DataSets['w0']
        self.DataSets['frame_height']=1060

        self.DataSets['low_green']=np.array([36, 25, 25])
        self.DataSets['high_green']=np.array([126, 225, 225])
        
        return self.DataSets

    def Data_31072020_7201st(self):

        self.DataSets['pitch']= str(30)
        self.DataSets['path']="/media/dom/Elements/data/TX2_Risehm/data_310720_1st/720p1st/goVR720p.avi"
        self.DataSets['row0']=100
        self.DataSets['col0']=40
        self.DataSets['colR']=940 # right side of column in original image
        self.DataSets['w0']=self.DataSets['colR'] - self.DataSets['col0']
        self.DataSets['lanebandwidth']=80

        self.DataSets['NumofLanes']=6
        self.DataSets['laneoffset']=90
        self.DataSets['LaneOriPoint']=30
        self.DataSets['frm2strart']=800

        self.DataSets['frame_width']=self.DataSets['w0']
        self.DataSets['frame_height']=1060

        self.DataSets['low_green']=np.array([36, 25, 25])
        self.DataSets['high_green']=np.array([126, 225, 225])
        
        return self.DataSets


    def Data_31072020_7202nd_720p1st(self):

        self.DataSets['pitch']= str(30)
        self.DataSets['path']="/media/dom/Elements/data/TX2_Risehm/data_310720_2nd/720q3rd/goVR720q.avi"
        self.DataSets['row0']=150
        self.DataSets['col0']=120
        self.DataSets['colR']=960 # right side of column in original image
        self.DataSets['w0']=self.DataSets['colR'] - self.DataSets['col0']
        self.DataSets['lanebandwidth']=80

        self.DataSets['NumofLanes']=6
        self.DataSets['laneoffset']=120
        self.DataSets['LaneOriPoint']=50
        self.DataSets['frm2strart']=130

        self.DataSets['frame_width']=self.DataSets['w0']
        self.DataSets['frame_height']=620

        self.DataSets['low_green']=np.array([36, 25, 25])
        self.DataSets['high_green']=np.array([126, 225, 225])
        
        return self.DataSets

    def Data_31072020_10802nd_1808p1st(self):

        self.DataSets['pitch']= str(30)
        self.DataSets['path']="/media/dom/Elements/data/TX2_Risehm/data_310720_2nd/1080p1st/goVR1080p.avi"
        self.DataSets['row0']=400
        self.DataSets['col0']=0
        self.DataSets['colR']=1500 # right side of column in original image
        self.DataSets['w0']=self.DataSets['colR'] - self.DataSets['col0']
        self.DataSets['lanebandwidth']=120

        self.DataSets['NumofLanes']=6
        self.DataSets['laneoffset']=20
        self.DataSets['LaneOriPoint']=50
        self.DataSets['frm2strart']=90

        self.DataSets['frame_width']=self.DataSets['w0']
        self.DataSets['frame_height']=620

        self.DataSets['low_green']=np.array([36, 25, 25])
        self.DataSets['high_green']=np.array([126, 225, 225])
        
        return self.DataSets


    ##################################################################
    # 07-08-2020 new wheat crops
    def data_07082020_5deg_new_720_1(self):

        self.DataSets['pitch']= '_0708_720q_'+str(5)
        self.DataSets['path']="/media/dom/Elements/data/TX2_Risehm/data_07082020_5deg_new/720q_1/goVR720q.avi"
        self.DataSets['path_dpt']="/media/dom/Elements/data/TX2_Risehm/data_07082020_5deg_new/720q_1/goVD720q.avi"

        self.DataSets['row0']=320
        self.DataSets['LaneOriPointL']=200
        self.DataSets['LaneEndPointR']=1000
        self.DataSets['frm2strart']=90#700#90#1600#900#90

        # self.DataSets['col0']=0
        # self.DataSets['colR']=960 # right side of column in original image
        self.DataSets['lanebandwidth']=80
        self.DataSets['NumofLanes']=6
        self.DataSets['laneoffset']=10


        # self.DataSets['w0']=self.DataSets['LaneEndPointR'] - self.DataSets['LaneOriPointL']+self.DataSets['lanebandwidth']
        # self.DataSets['frame_width']=self.DataSets['w0']
        self.DataSets['frame_height']=400
        # low_green = np.array([42, 25, 25])
        # high_green = np.array([106, 225, 225])

        # low_green = np.array([25, 52, 72])
        # high_green = np.array([102, 255, 255])

        # low_green = np.array([18, 94, 140])
        # high_green = np.array([48, 255, 255])

        # low_green = np.array([30, 0, 0])
        # high_green = np.array([100, 255, 255])

        self.DataSets['low_green']=np.array([36, 25, 25])
        self.DataSets['high_green']=np.array([126, 225, 225])
        
        return self.DataSets

    def data_07082020_30deg_new_720q_1(self):

        self.DataSets['pitch']= '0708_720q_'+str(30)
        self.DataSets['path']="/media/dom/Elements/data/TX2_Risehm/data_07082020_30deg_new/720q_1/goVL720q.avi"
        self.DataSets['row0']=200
        self.DataSets['col0']=120
        self.DataSets['colR']=960 # right side of column in original image
        self.DataSets['w0']=self.DataSets['colR'] - self.DataSets['col0']
        self.DataSets['lanebandwidth']=80

        self.DataSets['NumofLanes']=6
        self.DataSets['laneoffset']=25
        self.DataSets['LaneOriPoint']=100
        self.DataSets['frm2strart']=10

        self.DataSets['frame_width']=self.DataSets['w0']
        self.DataSets['frame_height']=300

        self.DataSets['low_green']=np.array([36, 25, 25])
        self.DataSets['high_green']=np.array([126, 225, 225])
        
        return self.DataSets

    def data_07082020_30deg_new_1080q_1(self):

        self.DataSets['pitch']= '_0708_1080q_'+str(30)
        self.DataSets['path']="/media/dom/Elements/data/TX2_Risehm/data_07082020_30deg_new/1080q_1/goVR1080q.avi"
        self.DataSets['row0']=300
        self.DataSets['col0']=0
        self.DataSets['colR']=1500 # right side of column in original image
        self.DataSets['w0']=self.DataSets['colR'] - self.DataSets['col0']
        self.DataSets['lanebandwidth']=120

        self.DataSets['NumofLanes']=6
        self.DataSets['laneoffset']=20
        self.DataSets['LaneOriPoint']=50
        self.DataSets['frm2strart']=90

        self.DataSets['frame_width']=self.DataSets['w0']
        self.DataSets['frame_height']=620

        self.DataSets['low_green']=np.array([36, 25, 25])
        self.DataSets['high_green']=np.array([126, 225, 225])
        
        return self.DataSets


    def data_18082020_85deg_720q(self):

        self.DataSets['pitch']= '_0708_1080q_'+str(30)
        self.DataSets['path']="/media/dom/Elements/data/data_18082020_2_manualctrl/720P85/goVL720p.avi"
        self.DataSets['path_dpt']="/media/dom/Elements/data/data_18082020_2_manualctrl/720P85/goVD720p.avi"

        self.DataSets['row0']=5
        self.DataSets['LaneOriPointL']=300
        self.DataSets['LaneEndPointR']=1020
        self.DataSets['frm2strart']=50#700#90#1600#900#90

        # self.DataSets['col0']=0
        # self.DataSets['colR']=960 # right side of column in original image
        self.DataSets['lanebandwidth']=80
        self.DataSets['NumofLanes']=6
        self.DataSets['laneoffset']=55


        # self.DataSets['w0']=self.DataSets['LaneEndPointR'] - self.DataSets['LaneOriPointL']+self.DataSets['lanebandwidth']
        # self.DataSets['frame_width']=self.DataSets['w0']
        self.DataSets['frame_height']=300
        # low_green = np.array([42, 25, 25])
        # high_green = np.array([106, 225, 225])

        # low_green = np.array([25, 52, 72])
        # high_green = np.array([102, 255, 255])

        # low_green = np.array([18, 94, 140])
        # high_green = np.array([48, 255, 255])

        # low_green = np.array([30, 0, 0])
        # high_green = np.array([100, 255, 255])

        self.DataSets['low_green']=np.array([36, 25, 25])
        self.DataSets['high_green']=np.array([126, 225, 225])
        
        return self.DataSets

    #####################################################################
    #####################################################################
    def data_02092020_deg_all(self):        
        
        self.DataSets['pitch']= '_0209_720q_'+str(75) 
        self.DataSets['path']="/media/dom/TOSHIBA EXT/data_02092020/720Q75/goVL720q.avi"
        self.DataSets['path_dpt']="/media/dom/TOSHIBA EXT/data_02092020/720Q75/goVD720q.avi"
        self.DataSets['row0']=0# 5 for detection
        self.DataSets['LaneOriPointL']=200#215##215#200
        self.DataSets['LaneEndPointR']=1200#1850#1100#1850
        self.DataSets['frm2strart']=140#-1#700#90#1600#900#90
        self.DataSets['laneoffset']=10#90#150#30#50
        self.DataSets['frame_height']=450
        """
     
        
        """
        self.DataSets['lanebandwidth']=80
        self.DataSets['NumofLanes']=6
        self.DataSets['low_green']=np.array([36, 25, 25])
        self.DataSets['high_green']=np.array([126, 225, 225])
        
        return self.DataSets
    #####################################################################
  
#####################################################################
    def data_09092020_deg_all(self):        
        """
        """
        self.DataSets['pitch']= '_0909_720q_'+str(80)
        self.DataSets['path']="/media/dom/TOSHIBA EXT/data_09092020/720p80_1/goVL720p.avi"       
        self.DataSets['path_dpt']="/media/dom/TOSHIBA EXT/data_09092020/720p80_1/goVD720p.avi"        

        self.DataSets['row0']=150# 5 for detection
        self.DataSets['LaneOriPointL']=250#215##215#200
        self.DataSets['LaneEndPointR']=1050#1850#1100#1850
        self.DataSets['frm2strart']=15#700#90#1600#900#90
        self.DataSets['laneoffset']=70#90#150#30#50
        self.DataSets['frame_height']=350#700#300#450# 300 for detection
        """       
        """


        # self.DataSets['frame_height']=600#700#300#450# 300 for detection

        self.DataSets['lanebandwidth']=80
        self.DataSets['NumofLanes']=6
        self.DataSets['low_green']=np.array([36, 25, 25])
        self.DataSets['high_green']=np.array([126, 225, 225])
        
        return self.DataSets
    #####################################################################
    def data_22102020_deg_all(self):        
        """
        """
        self.DataSets['pitch']= '_2010_720q_'+str(45)
        # self.DataSets['path']="/media/dom/TOSHIBA EXT/data_22102020/720q45deg120cm_0/goVL720q.avi"       
        # self.DataSets['path_dpt']="/media/dom/TOSHIBA EXT/data_22102020/720q45deg120cm_0/goVD720q.avi"   
        self.DataSets['path']="/media/dom/TOSHIBA EXT/data_22102020/720q45deg120cm_2/goVL720q.avi"       
        self.DataSets['path_dpt']="/media/dom/TOSHIBA EXT/data_22102020/720q45deg120cm_2/goVD720q.avi"       
        # 720q45deg120cm_0 : offset  =150
        # 720q45deg120cm_1 : offset  =200
        # 720q45deg120cm_2 : offset  =180
        self.DataSets['row0']=440# 5 for detection
        self.DataSets['LaneOriPointL']=280#215##215#200
        self.DataSets['LaneEndPointR']=1150#1850#1100#1850
        self.DataSets['frm2strart']=25#700#90#1600#900#90
        self.DataSets['laneoffset']=50#90#150#30#50
        self.DataSets['frame_height']=240#700#300#450# 300 for detection
        """       
        """


        # self.DataSets['frame_height']=600#700#300#450# 300 for detection

        self.DataSets['lanebandwidth']=80
        self.DataSets['NumofLanes']=6

        self.DataSets['low_green'] = np.array([20, 25,25])
        # self.DataSets['low_green'] = np.array([42, 0, 0])
        self.DataSets['high_green'] = np.array([100, 255, 255])

        # self.DataSets['low_green'] = np.array([25, 52, 72])
        # self.DataSets['high_green'] = np.array([102, 255, 255])

        # self.DataSets['low_green'] = np.array([18, 94, 140])
        # self.DataSets['high_green'] = np.array([48, 255, 255])

        # self.DataSets['low_green'] = np.array([30, 10, 10])
        # self.DataSets['high_green'] = np.array([100, 255, 255])

        # self.DataSets['low_green']=np.array([36, 25, 25])
        # self.DataSets['high_green']=np.array([126, 225, 225])
        
        return self.DataSets
    #####################################################################
    def data_05112020_4lanes_all(self): 
        """
        self.DataSets['pitch']= '_0511_720q_'+str(45)
        self.DataSets['path']="/media/dom/TOSHIBA EXT/data05112020/720q30_4/goVR720q.avi"       
        self.DataSets['path_dpt']="/media/dom/TOSHIBA EXT/data05112020/720q30_4/goVD720q.avi"

        # self.DataSets['path']="/media/dom/TOSHIBA EXT/data05112020/720q30_5/goVR720q.avi"       
        # self.DataSets['path_dpt']="/media/dom/TOSHIBA EXT/data05112020/720q30_5/goVD720q.avi" 

        self.DataSets['row0']=480# 5 for detection
        self.DataSets['LaneOriPointL']=340#215##215#200
        self.DataSets['LaneEndPointR']=830#1850#1100#1850
        self.DataSets['frm2strart']=25#700#90#1600#900#90
        self.DataSets['laneoffset']=25#90#150#30#50 
        self.DataSets['lanebandwidth']=80          
             
        
        self.DataSets['pitch']= '_0511_720q_'+str(45)
        self.DataSets['path']="/media/dom/TOSHIBA EXT/data05112020/720q45_2/goVR720q.avi"       
        self.DataSets['path_dpt']="/media/dom/TOSHIBA EXT/data05112020/720q45_2/goVD720q.avi"

        self.DataSets['path']="/media/dom/TOSHIBA EXT/data05112020/720q45_1/goVR720q.avi"       
        self.DataSets['path_dpt']="/media/dom/TOSHIBA EXT/data05112020/720q45_1/goVD720q.avi" 

        self.DataSets['row0']=480# 5 for detection
        self.DataSets['LaneOriPointL']=340#215##215#200
        self.DataSets['LaneEndPointR']=830#1850#1100#1850
        self.DataSets['frm2strart']=25#700#90#1600#900#90
        self.DataSets['laneoffset']=50#90#150#30#50 
        self.DataSets['lanebandwidth']=80              
       
        """
        self.DataSets['pitch']= '_0511_720q_'+str(45)
        self.DataSets['path']="/media/dom/TOSHIBA EXT/data05112020/720q45_3/goVL720q.avi"       
        self.DataSets['path_dpt']="/media/dom/TOSHIBA EXT/data05112020/720q45_3/goVD720q.avi"       
        self.DataSets['row0']=480# 5 for detection
        self.DataSets['LaneOriPointL']=340#215##215#200
        self.DataSets['LaneEndPointR']=830#1850#1100#1850
        self.DataSets['frm2strart']=25#700#90#1600#900#90
        self.DataSets['laneoffset']=60#90#150#30#50
        self.DataSets['lanebandwidth']=80
        
        """
        # with 15=pitch, and long covering FOV...
        self.DataSets['pitch']= '_0511_720q_'+str(15)
        self.DataSets['path']="/media/dom/TOSHIBA EXT/data05112020/720q15_6/goVL720q.avi"       
        self.DataSets['path_dpt']="/media/dom/TOSHIBA EXT/data05112020/720q15_6/goVD720q.avi"       
        self.DataSets['row0']=480# 5 for detection
        self.DataSets['LaneOriPointL']=400#215##215#200
        self.DataSets['LaneEndPointR']=825#1850#1100#1850
        self.DataSets['frm2strart']=25#700#90#1600#900#90
        self.DataSets['laneoffset']=140#90#150#30#50
        self.DataSets['lanebandwidth']=50
        ################################################
        """
        self.DataSets['frame_height']=240#700#300#450# 300 for detection      
        self.DataSets['NumofLanes']=4
        self.DataSets['low_green'] = np.array([20, 25,25])
        # self.DataSets['low_green'] = np.array([42, 0, 0])
        self.DataSets['high_green'] = np.array([100, 255, 255])

        # self.DataSets['low_green'] = np.array([25, 52, 72])
        # self.DataSets['high_green'] = np.array([102, 255, 255])

        # self.DataSets['low_green'] = np.array([18, 94, 140])
        # self.DataSets['high_green'] = np.array([48, 255, 255])

        # self.DataSets['low_green'] = np.array([30, 10, 10])
        # self.DataSets['high_green'] = np.array([100, 255, 255])

        # self.DataSets['low_green']=np.array([36, 25, 25])
        # self.DataSets['high_green']=np.array([126, 225, 225])
        
        return self.DataSets



 #####################################################################
    def data_12112020_4lanes_all(self): 
        """ 
        # with 45=pitch, and long covering FOV...            
        self.DataSets['pitch']= '_1211_720q_'+str(45)
        self.DataSets['path']="/media/dom/TOSHIBA EXT/data12112020/720q45_Peg2/goVL720q.avi"       
        self.DataSets['path_dpt']="/media/dom/TOSHIBA EXT/data12112020/720q45_Peg2/goVD720q.avi"       
        self.DataSets['row0']=400# 5 for detection
        self.DataSets['LaneOriPointL']=460#420
        self.DataSets['LaneEndPointR']=820#1850#1100#1850
        self.DataSets['frm2strart']=475#700#90#1600#900#90
        self.DataSets['laneoffset']=35#90#150#30#50
        self.DataSets['lanebandwidth']=35
        self.DataSets['frame_height']=240#700#300#450# 300 for detection 
        self.DataSets['NumofLanes']=5 
        ############################################################
         
        # with 30=pitch, and long covering FOV... not good 
        self.DataSets['pitch']= '_0511_720q_'+str(30)
        self.DataSets['path']="/media/dom/TOSHIBA EXT/data12112020/720q30_Peg3/goVL720q.avi"       
        self.DataSets['path_dpt']="/media/dom/TOSHIBA EXT/data12112020/720q30_Peg3/goVD720q.avi"       
        self.DataSets['row0']=420# 5 for detection
        self.DataSets['LaneOriPointL']=530#215##215#200
        self.DataSets['LaneEndPointR']=750#1850#1100#1850
        self.DataSets['frm2strart']=25#700#90#1600#900#90
        self.DataSets['laneoffset']=30#90#150#30#50
        self.DataSets['lanebandwidth']=35 
        self.DataSets['frame_height']=240#700#300#450# 300 for detection        
        self.DataSets['NumofLanes']=4 
        ##############################################################
        """
        """
        # with 30=pitch, and long covering FOV... show Charels , the harding...
        self.DataSets['pitch']= '_0511_720q_'+str(30)
        self.DataSets['path']="/media/dom/TOSHIBA EXT/data12112020/720q30_Peg4/goVL720q.avi"       
        self.DataSets['path_dpt']="/media/dom/TOSHIBA EXT/data12112020/720q30_Peg4/goVD720q.avi"       
        self.DataSets['row0']=400# 5 for detection
        self.DataSets['LaneOriPointL']=490#215##215#200
        self.DataSets['LaneEndPointR']=790#1850#1100#1850
        self.DataSets['frm2strart']=3#700#90#1600#900#90
        self.DataSets['laneoffset']=25#90#150#30#50
        self.DataSets['lanebandwidth']=35  
        self.DataSets['frame_height']=240#700#300#450# 300 for detection 
        self.DataSets['NumofLanes']=5
        ###############################################################
        """ 
        """
        # with 30=pitch, and long covering FOV...  this one can used for trainng only
        self.DataSets['pitch']= '_0511_720q_'+str(30)
        self.DataSets['path']="/media/dom/TOSHIBA EXT/data12112020/720q30_Peg5/goVL720q.avi"       
        self.DataSets['path_dpt']="/media/dom/TOSHIBA EXT/data12112020/720q30_Peg5/goVD720q.avi"       
        self.DataSets['row0']=400# 5 for detection
        self.DataSets['LaneOriPointL']=470#215##215#200
        self.DataSets['LaneEndPointR']=810#1850#1100#1850
        self.DataSets['frm2strart']=68#700#90#1600#900#90
        self.DataSets['laneoffset']=35#90#150#30#50
        self.DataSets['lanebandwidth']=40 
        self.DataSets['frame_height']=240#700#300#450# 300 for detection 
        self.DataSets['NumofLanes']=5  
        
        ################################################################
        # with 45=pitch, and long covering FOV...  shadow . light darker 
        self.DataSets['pitch']= '_0511_720q_'+str(45)
        self.DataSets['path']="/media/dom/TOSHIBA EXT/data12112020/720q45_Peg1/goVL720q.avi"       
        self.DataSets['path_dpt']="/media/dom/TOSHIBA EXT/data12112020/720q45_Peg1/goVD720q.avi"       
        self.DataSets['row0']=400# 5 for detection
        self.DataSets['LaneOriPointL']=440#215##215#200
        self.DataSets['LaneEndPointR']=840#1850#1100#1850
        self.DataSets['frm2strart']=3#700#90#1600#900#90
        self.DataSets['laneoffset']=25#90#150#30#50
        self.DataSets['lanebandwidth']=35 
        self.DataSets['frame_height']=240#700#300#450# 300 for detection  
        self.DataSets['NumofLanes']=6 
        ###########################################################################
        """
        """
        # with 45=pitch, and long covering FOV...  light  
        self.DataSets['pitch']= '_0511_720q_'+str(45)
        self.DataSets['path']="/media/dom/TOSHIBA EXT/data12112020/720q45_Peg2/goVL720q.avi"       
        self.DataSets['path_dpt']="/media/dom/TOSHIBA EXT/data12112020/720q45_Peg2/goVD720q.avi"       
        self.DataSets['row0']=400# 5 for detection
        self.DataSets['LaneOriPointL']=450#215##215#200
        self.DataSets['LaneEndPointR']=830#1850#1100#1850
        self.DataSets['frm2strart']=500#700#90#1600#900#90
        self.DataSets['laneoffset']=35#90#150#30#50
        self.DataSets['lanebandwidth']=35  
        self.DataSets['frame_height']=240#700#300#450# 300 for detection 
        self.DataSets['NumofLanes']=6 
        """
        """
        ###########################################################################
        # with 45=pitch, and long covering FOV...  light  
        self.DataSets['pitch']= '_0511_720q_'+str(45)
        self.DataSets['path']="/media/dom/TOSHIBA EXT/data12112020/720q45_Peg3/goVL720q.avi"       
        self.DataSets['path_dpt']="/media/dom/TOSHIBA EXT/data12112020/720q45_Peg3/goVD720q.avi"       
        self.DataSets['row0']=400# 5 for detection
        self.DataSets['LaneOriPointL']=440#215##215#200
        self.DataSets['LaneEndPointR']=840#1850#1100#1850
        self.DataSets['frm2strart']=275#700#90#1600#900#90
        self.DataSets['laneoffset']=35#90#150#30#50
        self.DataSets['lanebandwidth']=35 
        self.DataSets['frame_height']=240#700#300#450# 300 for detection  
        self.DataSets['NumofLanes']=6 
        
        ###########################################################################
        # with 45=pitch, and long covering FOV...  light  
        self.DataSets['pitch']= '_0511_720q_'+str(60)
        self.DataSets['path']="/media/dom/TOSHIBA EXT/data12112020/720q60_Peg1/goVL720q.avi"       
        self.DataSets['path_dpt']="/media/dom/TOSHIBA EXT/data12112020/720q60_Peg1/goVD720q.avi"       
        self.DataSets['row0']=400# 5 for detection
        self.DataSets['LaneOriPointL']=480#215##215#200
        self.DataSets['LaneEndPointR']=800#1850#1100#1850
        self.DataSets['frm2strart']=265#700#90#1600#900#90
        self.DataSets['laneoffset']=35#90#150#30#50
        self.DataSets['lanebandwidth']=35  
        self.DataSets['frame_height']=240#700#300#450# 300 for detection 
        self.DataSets['NumofLanes']=5  
              
        ###########################################################################
        # with 45=pitch, and long covering FOV...  shadow , darker  
        self.DataSets['pitch']= '_0511_720q_'+str(60)
        self.DataSets['path']="/media/dom/TOSHIBA EXT/data12112020/720q60_Peg2/goVL720q.avi"       
        self.DataSets['path_dpt']="/media/dom/TOSHIBA EXT/data12112020/720q60_Peg2/goVD720q.avi"       
        self.DataSets['row0']=400# 5 for detection
        self.DataSets['LaneOriPointL']=480#215##215#200
        self.DataSets['LaneEndPointR']=800#1850#1100#1850
        self.DataSets['frm2strart']=150#700#90#1600#900#90
        self.DataSets['laneoffset']=30#90#150#30#50
        self.DataSets['lanebandwidth']=25  
        self.DataSets['frame_height']=240#700#300#450# 300 for detection  
        self.DataSets['NumofLanes']=5
        ################################################
        """
        """
        ###########################################################################
        # with 5=pitch, and long covering FOV...  parrallet to the ground , .hard to line up at beging..
        self.DataSets['pitch']= '_0511_720q_'+str(5)
        self.DataSets['path']="/media/dom/TOSHIBA EXT/data12112020/720q5_noPeg1/goVL720q.avi"       
        self.DataSets['path_dpt']="/media/dom/TOSHIBA EXT/data12112020/720q5_noPeg1/goVD720q.avi"       
        self.DataSets['row0']=576# 5 for detection
        self.DataSets['LaneOriPointL']=540#215##215#200
        self.DataSets['LaneEndPointR']=740#1850#1100#1850
        self.DataSets['frm2strart']=50#700#90#1600#900#90
        self.DataSets['laneoffset']=50#90#150#30#50
        self.DataSets['lanebandwidth']=30  
        self.DataSets['frame_height']=144#700#300#450# 300 for detection
        self.DataSets['NumofLanes']=5  
        
        """
        """
        ####### this one can be used for domnos - 3 lanes only -   for Shaun and Charles
        # with 15=pitch, and long covering FOV...  hard to line up on the ground , ...
        self.DataSets['pitch']= '_0511_720q_'+str(15)
        self.DataSets['path']="/media/dom/TOSHIBA EXT/data12112020/720q15_noPeg1/goVL720q.avi"       
        self.DataSets['path_dpt']="/media/dom/TOSHIBA EXT/data12112020/720q15_noPeg1/goVD720q.avi"       
        self.DataSets['row0']=480# 5 for detection
        self.DataSets['LaneOriPointL']=520#215##215#200
        self.DataSets['LaneEndPointR']=780#1850#1100#1850
        self.DataSets['frm2strart']=250#700#90#1600#900#90
        self.DataSets['laneoffset']=55#90#150#30#50
        self.DataSets['lanebandwidth']=40  
        self.DataSets['frame_height']=240#700#300#450# 300 for detection 
        self.DataSets['NumofLanes']=3 
        ################################################
        """
       
        """
        ########### good fro demon to xxxxx
        # with 15=pitch, and long covering FOV...  hard to line up on the ground , ...
        self.DataSets['pitch']= '_0511_720q_'+str(30)
        self.DataSets['path']="/media/dom/TOSHIBA EXT/data12112020/720q30_noPeg1/goVL720q.avi"       
        self.DataSets['path_dpt']="/media/dom/TOSHIBA EXT/data12112020/720q30_noPeg1/goVD720q.avi"       
        self.DataSets['row0']=480# 5 for detection
        self.DataSets['LaneOriPointL']=520#215##215#200
        self.DataSets['LaneEndPointR']=760#1850#1100#1850
        self.DataSets['frm2strart']=50#700#90#1600#900#90
        self.DataSets['laneoffset']=40#90#150#30#50
        self.DataSets['lanebandwidth']=35  
        self.DataSets['frame_height']=240#700#300#450# 300 for detection 
        self.DataSets['NumofLanes']=4 
        ################################################
        """
        """
        # now good one 
        # with 30=pitch, and long covering FOV...  hard to line up on the ground , ...
        self.DataSets['pitch']= '_0511_720q_'+str(30)
        self.DataSets['path']="/media/dom/TOSHIBA EXT/data12112020/720q30_noPeg2/goVL720q.avi"       
        self.DataSets['path_dpt']="/media/dom/TOSHIBA EXT/data12112020/720q30_noPeg2/goVD720q.avi"       
        self.DataSets['row0']=480# 5 for detection
        self.DataSets['LaneOriPointL']=490#215##215#200
        self.DataSets['LaneEndPointR']=790#1850#1100#1850
        self.DataSets['frm2strart']=15#700#90#1600#900#90
        self.DataSets['laneoffset']=45#90#150#30#50
        self.DataSets['lanebandwidth']=40  
        self.DataSets['frame_height']=240#700#300#450# 300 for detection 
        self.DataSets['NumofLanes']=4 
        
        ### may be OK as well
        # with 30=pitch, and long covering FOV...low hieght 0.5meter, shadow, ...
        self.DataSets['pitch']= '_0511_720q_'+str(30)
        self.DataSets['path']="/media/dom/TOSHIBA EXT/data12112020/720q30_noPeg05m/goVL720q.avi"       
        self.DataSets['path_dpt']="/media/dom/TOSHIBA EXT/data12112020/720q30_noPeg05m/goVD720q.avi"       
        self.DataSets['row0']=330# 5 for detection
        self.DataSets['LaneOriPointL']=480#215##215#200
        self.DataSets['LaneEndPointR']=800#1850#1100#1850
        self.DataSets['frm2strart']=15#700#90#1600#900#90
        self.DataSets['laneoffset']=50#90#150#30#50
        self.DataSets['lanebandwidth']=45  
        self.DataSets['frame_height']=240#700#300#450# 300 for detection 
        self.DataSets['NumofLanes']=4 
        
        #######################################################
        # 
        # with 45=pitch, and long covering FOV...low hieght , ...
        self.DataSets['pitch']= '_0511_720q_'+str(45)
        self.DataSets['path']="/media/dom/TOSHIBA EXT/data12112020/720q45_noPeg1/goVL720q.avi"       
        self.DataSets['path_dpt']="/media/dom/TOSHIBA EXT/data12112020/720q45_noPeg1/goVD720q.avi"       
        self.DataSets['row0']=480# 5 for detection
        self.DataSets['LaneOriPointL']=450#360##215#200
        self.DataSets['LaneEndPointR']=830#1850#1100#1850
        self.DataSets['frm2strart']=15#700#90#1600#900#90
        self.DataSets['laneoffset']=45#90#150#30#50
        self.DataSets['lanebandwidth']=35  
        self.DataSets['frame_height']=240#700#300#450# 300 for detection 
        self.DataSets['NumofLanes']=5 
        """
        """        
        # No good with 55=pitch, and long covering FOV..not good because low hieght, distortation uch wiht big pithch and lower distannce to ground, ...
        self.DataSets['pitch']= '_0511_720q_'+str(55)
        self.DataSets['path']="/media/dom/TOSHIBA EXT/data12112020/720q55_noPeg1/goVL720q.avi"       
        self.DataSets['path_dpt']="/media/dom/TOSHIBA EXT/data12112020/720q55_noPeg1/goVD720q.avi"       
        self.DataSets['row0']=430# 5 for detection
        self.DataSets['LaneOriPointL']=460#360##420#200
        self.DataSets['LaneEndPointR']=820#1850#1100#1850
        self.DataSets['frm2strart']=10#700#90#1600#900#90
        self.DataSets['laneoffset']=40#90#150#30#50
        self.DataSets['lanebandwidth']=35  
        self.DataSets['frame_height']=240#700#300#450# 300 for detection 
        self.DataSets['NumofLanes']=5 
        
        ###############################################
        
        # Not realy, with 75=pitch, and long covering FOV...leaves falling , very messy, and not good challenge
        self.DataSets['pitch']= '_0511_720q_'+str(75)
        self.DataSets['path']="/media/dom/TOSHIBA EXT/data12112020/720q75_noPeg1/goVL720q.avi"       
        self.DataSets['path_dpt']="/media/dom/TOSHIBA EXT/data12112020/720q75_noPeg1/goVD720q.avi"       
        self.DataSets['row0']=420# 5 for detection
        self.DataSets['LaneOriPointL']=500#360##420#480
        self.DataSets['LaneEndPointR']=780#1850#1100#1850
        self.DataSets['frm2strart']=10#700#90#1600#900#90
        self.DataSets['laneoffset']=30#90#150#30#50
        self.DataSets['lanebandwidth']=30  
        self.DataSets['frame_height']=240#700#300#450# 300 for detection 
        self.DataSets['NumofLanes']=5 
        
        ###############################################
        # not realy , can try with 85 =pitch, and long covering FOV...annce to ground, very messy, and not good challenge
        self.DataSets['pitch']= '_0511_720q_'+str(85)
        self.DataSets['path']="/media/dom/TOSHIBA EXT/data12112020/720q85_noPeg1/goVL720q.avi"       
        self.DataSets['path_dpt']="/media/dom/TOSHIBA EXT/data12112020/720q85_noPeg1/goVD720q.avi"       
        self.DataSets['row0']=420# 5 for detection
        self.DataSets['LaneOriPointL']=510#360##420#480
        self.DataSets['LaneEndPointR']=770#1850#1100#1850
        self.DataSets['frm2strart']=10#700#90#1600#900#90
        self.DataSets['laneoffset']=30#90#150#30#50
        self.DataSets['lanebandwidth']=30  
        self.DataSets['frame_height']=240#700#300#450# 300 for detection 
        self.DataSets['NumofLanes']=4 
        
        
        ###############################################
        # with 85 =pitch, and long covering FOV...annce to ground, very messy, and not good challenge
        self.DataSets['pitch']= '_0511_720q_'+str(85)
        self.DataSets['path']="/media/dom/TOSHIBA EXT/data12112020/720q85_noPeg2/goVL720q.avi"       
        self.DataSets['path_dpt']="/media/dom/TOSHIBA EXT/data12112020/720q85_noPeg2/goVD720q.avi"       
        self.DataSets['row0']=420# 5 for detection
        self.DataSets['LaneOriPointL']=480#360##420#480
        self.DataSets['LaneEndPointR']=800#1850#1100#1850
        self.DataSets['frm2strart']=10#700#90#1600#900#90
        self.DataSets['laneoffset']=25#90#150#30#50
        self.DataSets['lanebandwidth']=35  
        self.DataSets['frame_height']=240#700#300#450# 300 for detection 
        self.DataSets['NumofLanes']=5         
        """

        """
        #########for statistic nalysis ################################
        # looks OK at beging.  by far
        # with 45=pitch, and long covering FOV...annce to ground, ...
        self.DataSets['pitch']= '_0511_720q_'+str(45)
        self.DataSets['path']="/media/dom/TOSHIBA EXT/data12112020/720q45_noPeg2_c3/goVL720q.avi"       
        self.DataSets['path_dpt']="/media/dom/TOSHIBA EXT/data12112020/720q45_noPeg2_c3/goVD720q.avi"       
        self.DataSets['row0']=430# 5 for detection
        self.DataSets['LaneOriPointL']=500#360##420#480
        self.DataSets['LaneEndPointR']=780#1850#1100#1850
        self.DataSets['frm2strart']=10#700#90#1600#900#90
        self.DataSets['laneoffset']=30#90#150#30#50
        self.DataSets['lanebandwidth']=25  
        self.DataSets['frame_height']=240#700#300#450# 300 for detection 
        self.DataSets['NumofLanes']=5 
        """ 
        ###########for statistic nalysis############################
        # good with 60=pitch, and long covering FOV...annce to ground, good...
        # pappwriint dataset 1  gap = 12.5 cm , pixels = 800-480/(5-1) = 80
        self.DataSets['pitch']= '720q60_noPeg1_c2'+str(60)
        self.DataSets['path']="/media/dom/TOSHIBA EXT/data12112020/720q60_noPeg1_c2/goVL720q.avi"       
        self.DataSets['path_dpt']="/media/dom/TOSHIBA EXT/data12112020/720q60_noPeg1_c2/goVD720q.avi"       
        self.DataSets['row0']=420# 5 for detection
        self.DataSets['LaneOriPointL']=480#360##420#480
        self.DataSets['LaneEndPointR']=800#1850#1100#1850
        self.DataSets['frm2start']=10#700#90#1600#900#90
        self.DataSets['laneoffset']=35#90#150#30#50
        self.DataSets['lanebandwidth']=35  
        self.DataSets['frame_height']=240#700#300#450# 300 for detection 
        self.DataSets['NumofLanes']=5  
        
        """        
        ###############for statistic nalysis#################
        # good with 60=pitch, and long covering FOV...annce to ground, good... low height
        self.DataSets['pitch']= '_0511_720q_'+str(60)
        self.DataSets['path']="/media/dom/TOSHIBA EXT/data12112020/720q60_noPeg2_c1/goVL720q.avi"       
        self.DataSets['path_dpt']="/media/dom/TOSHIBA EXT/data12112020/720q60_noPeg2_c1/goVD720q.avi"       
        self.DataSets['row0']=420# 5 for detection
        self.DataSets['LaneOriPointL']=510#360##420#480
        self.DataSets['LaneEndPointR']=770#1850#1100#1850
        self.DataSets['frm2strart']=3#700#90#1600#900#90
        self.DataSets['laneoffset']=30#90#150#30#50
        self.DataSets['lanebandwidth']=35  
        self.DataSets['frame_height']=240#700#300#450# 300 for detection 
        self.DataSets['NumofLanes']=4 
        """ 
        ###############################################
        self.DataSets['low_green'] = np.array([20, 25,25])
        # self.DataSets['low_green'] = np.array([42, 0, 0])
        self.DataSets['high_green'] = np.array([100, 255, 255])

        # self.DataSets['low_green'] = np.array([25, 52, 72])
        # self.DataSets['high_green'] = np.array([102, 255, 255])

        # self.DataSets['low_green'] = np.array([18, 94, 140])
        # self.DataSets['high_green'] = np.array([48, 255, 255])

        # self.DataSets['low_green'] = np.array([30, 10, 10])
        # self.DataSets['high_green'] = np.array([100, 255, 255])

        # self.DataSets['low_green']=np.array([36, 25, 25])
        # self.DataSets['high_green']=np.array([126, 225, 225])
        
        return self.DataSets
    #####################################################################
    #####################################################################
    def data_19112020_4lanes_all(self): 
        """
        # with 45=pitch, and long covering FOV...            
        self.DataSets['pitch']= '_1911_720q_'+str(45)
        self.DataSets['path']="/media/dom/TOSHIBA EXT/data19112020/720p45_shdow4/goVL720q.avi"       
        self.DataSets['path_dpt']="/media/dom/TOSHIBA EXT/data19112020/720p45_shdow4/goVD720q.avi"       
        self.DataSets['row0']=480# 5 for detection
        self.DataSets['LaneOriPointL']=480#420
        self.DataSets['LaneEndPointR']=800#1850#1100#1850
        self.DataSets['frm2strart']=2#700#90#1600#900#90
        self.DataSets['laneoffset']=60#90#150#30#50
        self.DataSets['lanebandwidth']=35
        self.DataSets['frame_height']=240#700#300#450# 300 for detection 
        self.DataSets['NumofLanes']=5
        ############################################################
       
        # with 30=pitch, and long covering FOV... not good 
        self.DataSets['pitch']= '_1911_720p_'+str(60)
        self.DataSets['path']="/media/dom/TOSHIBA EXT/data19112020/720p60_wide2/goVL720p.avi"       
        self.DataSets['path_dpt']="/media/dom/TOSHIBA EXT/data19112020/720p60_wide2/goVD720p.avi"       
        self.DataSets['row0']=480# 5 for detection
        self.DataSets['LaneOriPointL']=470#215##215#200
        self.DataSets['LaneEndPointR']=810#1850#1100#1850
        self.DataSets['frm2strart']=5#700#90#1600#900#90
        self.DataSets['laneoffset']=35#90#150#30#50
        self.DataSets['lanebandwidth']=60 
        self.DataSets['frame_height']=240#700#300#450# 300 for detection        
        self.DataSets['NumofLanes']=3 
        ##############################################################        
        
        # with 30=pitch, and long covering FOV... show Charels , the harding...
        self.DataSets['pitch']= '_1911_720q_'+str(5)
        self.DataSets['path']="/media/dom/TOSHIBA EXT/data19112020/720q5_verylow/goVL720q.avi"       
        self.DataSets['path_dpt']="/media/dom/TOSHIBA EXT/data19112020/720q5_verylow/goVD720q.avi"       
        self.DataSets['row0']=480# 5 for detection
        self.DataSets['LaneOriPointL']=440#215##215#200
        self.DataSets['LaneEndPointR']=840#1850#1100#1850
        self.DataSets['frm2start']=15#700#90#1600#900#90
        self.DataSets['laneoffset']=110#90#150#30#50
        self.DataSets['lanebandwidth']=80  
        self.DataSets['frame_height']=240#700#300#450# 300 for detection 
        self.DataSets['NumofLanes']=3
        ###############################################################
               
        # with 30=pitch, and long covering FOV...  this one can used for trainng only
        self.DataSets['pitch']= '_1911_720q_'+str(45)
        self.DataSets['path']="/media/dom/TOSHIBA EXT/data19112020/720q45_shwd5/goVL720q.avi"       
        self.DataSets['path_dpt']="/media/dom/TOSHIBA EXT/data19112020/720q45_shwd5/goVD720q.avi"       
        self.DataSets['row0']=480# 5 for detection
        self.DataSets['LaneOriPointL']=460#215##215#200
        self.DataSets['LaneEndPointR']=760#1850#1100#1850
        self.DataSets['frm2start']=5#700#90#1600#900#90
        self.DataSets['laneoffset']=40#90#150#30#50
        self.DataSets['lanebandwidth']=30 
        self.DataSets['frame_height']=240#700#300#450# 300 for detection 
        self.DataSets['NumofLanes']=5  
        
        ################################################################
        # with 60=pitch, and long covering FOV...  shadow . light darker 
        self.DataSets['pitch']= '_1911_720q_'+str(60)
        self.DataSets['path']="/media/dom/TOSHIBA EXT/data19112020/720q60_1/goVL720q.avi"       
        self.DataSets['path_dpt']="/media/dom/TOSHIBA EXT/data19112020/720q60_1/goVD720q.avi"       
        self.DataSets['row0']=480# 5 for detection
        self.DataSets['LaneOriPointL']=500#215##215#200
        self.DataSets['LaneEndPointR']=780#1850#1100#1850
        self.DataSets['frm2start']=5#700#90#1600#900#90
        self.DataSets['laneoffset']=20#90#150#30#50
        self.DataSets['lanebandwidth']=35 
        self.DataSets['frame_height']=240#700#300#450# 300 for detection  
        self.DataSets['NumofLanes']=5 
        ###########################################################################
        """
        # with 60=pitch, and long covering FOV...  light  
        # pape writing for dataset2 gap = 15 cm, = 760-520/(4-1)= 80
        self.DataSets['pitch']= '720q60_3_c1'+str(60)
        self.DataSets['path']="/media/dom/TOSHIBA EXT/data19112020/720q60_3_c1/goVL720q.avi"       
        self.DataSets['path_dpt']="/media/dom/TOSHIBA EXT/data19112020/720q60_3_c1/goVD720q.avi"       
        self.DataSets['row0']=480# 5 for detection
        self.DataSets['LaneOriPointL']=520#215##215#200
        self.DataSets['LaneEndPointR']=760#1850#1100#1850
        self.DataSets['frm2start']=5#700#90#1600#900#90
        self.DataSets['laneoffset']=30#90#150#30#50
        self.DataSets['lanebandwidth']=35  
        self.DataSets['frame_height']=240#700#300#450# 300 for detection 
        self.DataSets['NumofLanes']=4         
        """
        ###########################################################################
        # with 60=pitch, and long covering FOV...  light  
        self.DataSets['pitch']= '_1911_720q_'+str(60)
        self.DataSets['path']="/media/dom/TOSHIBA EXT/data19112020/720q60_4/goVL720q.avi"       
        self.DataSets['path_dpt']="/media/dom/TOSHIBA EXT/data19112020/720q60_4/goVD720q.avi"       
        self.DataSets['row0']=480# 5 for detection
        self.DataSets['LaneOriPointL']=525#215##215#200
        self.DataSets['LaneEndPointR']=755#1850#1100#1850
        self.DataSets['frm2start']=5#700#90#1600#900#90
        self.DataSets['laneoffset']=25#90#150#30#50
        self.DataSets['lanebandwidth']=35 
        self.DataSets['frame_height']=240#700#300#450# 300 for detection  
        self.DataSets['NumofLanes']=4 
        
        ###########################################################################
        # with 45=pitch, and long covering FOV...  light  
        self.DataSets['pitch']= '_1911_720q_'+str(45)
        self.DataSets['path']="/media/dom/TOSHIBA EXT/data19112020/720q45_5/goVL720q.avi"       
        self.DataSets['path_dpt']="/media/dom/TOSHIBA EXT/data19112020/720q45_5/goVD720q.avi"       
        self.DataSets['row0']=480# 5 for detection
        self.DataSets['LaneOriPointL']=530#215##215#200
        self.DataSets['LaneEndPointR']=750#1850#1100#1850
        self.DataSets['frm2start']=5#700#90#1600#900#90
        self.DataSets['laneoffset']=25#90#150#30#50
        self.DataSets['lanebandwidth']=30 
        self.DataSets['frame_height']=240#700#300#450# 300 for detection  
        self.DataSets['NumofLanes']=4 
            
        ###########################################################################
        # with 60=pitch, and long covering FOV...  light  very low about 0.5m not good
        self.DataSets['pitch']= '_1911_720q_'+str(60)
        self.DataSets['path']="/media/dom/TOSHIBA EXT/data19112020/720q60_low/goVL720q.avi"       
        self.DataSets['path_dpt']="/media/dom/TOSHIBA EXT/data19112020/720q60_low/goVD720q.avi"       
        self.DataSets['row0']=380# 5 for detection
        self.DataSets['LaneOriPointL']=440#215##215#200
        self.DataSets['LaneEndPointR']=840#1850#1100#1850
        self.DataSets['frm2start']=5#700#90#1600#900#90
        self.DataSets['laneoffset']=40#90#150#30#50
        self.DataSets['lanebandwidth']=50 
        self.DataSets['frame_height']=240#700#300#450# 300 for detection  
        self.DataSets['NumofLanes']=4 
        ################################################
        
        """
        ###########################################################################
        # with 60=pitch, and long covering FOV...  light  very low about 0.5m not good
        self.DataSets['pitch']= '_1911_720q_'+str(60)
        self.DataSets['path']="/media/dom/TOSHIBA EXT/data19112020/720q60_shdow2_peg_c2/goVL720q.avi"       
        self.DataSets['path_dpt']="/media/dom/TOSHIBA EXT/data19112020/720q60_shdow2_peg_c2/goVD720q.avi"       
        self.DataSets['row0']=480# 5 for detection
        self.DataSets['LaneOriPointL']=550#215##215#200
        self.DataSets['LaneEndPointR']=730#1850#1100#1850
        self.DataSets['frm2start']=5#700#90#1600#900#90
        self.DataSets['laneoffset']=20#90#150#30#50
        self.DataSets['lanebandwidth']=35 
        self.DataSets['frame_height']=240#700#300#450# 300 for detection  
        self.DataSets['NumofLanes']=4 
        """
        
        
        ####### this one can be used for domnos - 3 lanes only -   for Shaun and Charles
        # with 60=pitch, and long covering FOV...  light  very low about 0.5m not good
        self.DataSets['pitch']= '_1911_720q_'+str(60)
        self.DataSets['path']="/media/dom/TOSHIBA EXT/data19112020/720q60_shdow4_wide/goVL720q.avi"       
        self.DataSets['path_dpt']="/media/dom/TOSHIBA EXT/data19112020/720q60_shdow4_wide/goVD720q.avi"       
        self.DataSets['row0']=480# 5 for detection
        self.DataSets['LaneOriPointL']=420#215##215#200
        self.DataSets['LaneEndPointR']=860#1850#1100#1850
        self.DataSets['frm2start']=5#700#90#1600#900#90
        self.DataSets['laneoffset']=50#90#150#30#50
        self.DataSets['lanebandwidth']=50 
        self.DataSets['frame_height']=240#700#300#450# 300 for detection  
        self.DataSets['NumofLanes']=4 
        ##########################################    
               
        ########### fro demon to xxxxx
        self.DataSets['pitch']= '_1911_720q_'+str(75)
        self.DataSets['path']="/media/dom/TOSHIBA EXT/data19112020/720q60_shdow_low/goVL720q.avi"       
        self.DataSets['path_dpt']="/media/dom/TOSHIBA EXT/data19112020/720q60_shdow_low/goVD720q.avi"       
        self.DataSets['row0']=400# 5 for detection
        self.DataSets['LaneOriPointL']=540#215##215#200
        self.DataSets['LaneEndPointR']=740#1850#1100#1850
        self.DataSets['frm2start']=5#700#90#1600#900#90
        self.DataSets['laneoffset']=20#90#150#30#50
        self.DataSets['lanebandwidth']=35 
        self.DataSets['frame_height']=240#700#300#450# 300 for detection  
        self.DataSets['NumofLanes']=4 
        ################################################
        
        # wide lane , not bad one
        self.DataSets['pitch']= '_1911_720q_'+str(60)
        self.DataSets['path']="/media/dom/TOSHIBA EXT/data19112020/720q60_wide1/goVL720q.avi"       
        self.DataSets['path_dpt']="/media/dom/TOSHIBA EXT/data19112020/720q60_wide1/goVD720q.avi"       
        self.DataSets['row0']=480# 5 for detection
        self.DataSets['LaneOriPointL']=380#215##215#200
        self.DataSets['LaneEndPointR']=900#1850#1100#1850
        self.DataSets['frm2start']=5#700#90#1600#900#90
        self.DataSets['laneoffset']=70#90#150#30#50
        self.DataSets['lanebandwidth']=60 
        self.DataSets['frame_height']=240#700#300#450# 300 for detection  
        self.DataSets['NumofLanes']=4 
        
        # low height, and bit pitch = 80
        self.DataSets['pitch']= '_1911_720q_'+str(80)
        self.DataSets['path']="/media/dom/TOSHIBA EXT/data19112020/720q80_low1/goVL720q.avi"       
        self.DataSets['path_dpt']="/media/dom/TOSHIBA EXT/data19112020/720q80_low1/goVD720q.avi"       
        self.DataSets['row0']=260# 5 for detection
        self.DataSets['LaneOriPointL']=480#215##215#200
        self.DataSets['LaneEndPointR']=800#1850#1100#1850
        self.DataSets['frm2start']=15#700#90#1600#900#90
        self.DataSets['laneoffset']=30#90#150#30#50
        self.DataSets['lanebandwidth']=35 
        self.DataSets['frame_height']=240#700#300#450# 300 for detection  
        self.DataSets['NumofLanes']=5         
        
        #######################################################
        # low, shadow, wide lane , big angel................
        self.DataSets['pitch']= '_1911_720q_'+str(80)
        self.DataSets['path']="/media/dom/TOSHIBA EXT/data19112020/720q80_wide_shdow/goVL720q.avi"       
        self.DataSets['path_dpt']="/media/dom/TOSHIBA EXT/data19112020/720q80_wide_shdow/goVD720q.avi"       
        self.DataSets['row0']=270# 5 for detection
        self.DataSets['LaneOriPointL']=290#215##215#200
        self.DataSets['LaneEndPointR']=910#1850#1100#1850
        self.DataSets['frm2start']=15#700#90#1600#900#90
        self.DataSets['laneoffset']=70#90#150#30#50
        self.DataSets['lanebandwidth']=60 
        self.DataSets['frame_height']=240#700#300#450# 300 for detection  
        self.DataSets['NumofLanes']=4 
        """
        """        
        # No good with 55=pitch, and long covering FOV..not good because low hieght, distortation uch wiht big pithch and lower distannce to ground, ...
        self.DataSets['pitch']= '_0511_720q_'+str(55)
        self.DataSets['path']="/media/dom/TOSHIBA EXT/data12112020/720q55_noPeg1/goVL720q.avi"       
        self.DataSets['path_dpt']="/media/dom/TOSHIBA EXT/data12112020/720q55_noPeg1/goVD720q.avi"       
        self.DataSets['row0']=430# 5 for detection
        self.DataSets['LaneOriPointL']=460#360##420#200
        self.DataSets['LaneEndPointR']=820#1850#1100#1850
        self.DataSets['frm2start']=10#700#90#1600#900#90
        self.DataSets['laneoffset']=40#90#150#30#50
        self.DataSets['lanebandwidth']=35  
        self.DataSets['frame_height']=240#700#300#450# 300 for detection 
        self.DataSets['NumofLanes']=5 
        
        ###############################################
        
        # Not realy, with 75=pitch, and long covering FOV...leaves falling , very messy, and not good challenge
        self.DataSets['pitch']= '_0511_720q_'+str(75)
        self.DataSets['path']="/media/dom/TOSHIBA EXT/data12112020/720q75_noPeg1/goVL720q.avi"       
        self.DataSets['path_dpt']="/media/dom/TOSHIBA EXT/data12112020/720q75_noPeg1/goVD720q.avi"       
        self.DataSets['row0']=420# 5 for detection
        self.DataSets['LaneOriPointL']=500#360##420#480
        self.DataSets['LaneEndPointR']=780#1850#1100#1850
        self.DataSets['frm2start']=10#700#90#1600#900#90
        self.DataSets['laneoffset']=30#90#150#30#50
        self.DataSets['lanebandwidth']=30  
        self.DataSets['frame_height']=240#700#300#450# 300 for detection 
        self.DataSets['NumofLanes']=5 
        
        ###############################################
        # not realy , can try with 85 =pitch, and long covering FOV...annce to ground, very messy, and not good challenge
        self.DataSets['pitch']= '_0511_720q_'+str(85)
        self.DataSets['path']="/media/dom/TOSHIBA EXT/data12112020/720q85_noPeg1/goVL720q.avi"       
        self.DataSets['path_dpt']="/media/dom/TOSHIBA EXT/data12112020/720q85_noPeg1/goVD720q.avi"       
        self.DataSets['row0']=420# 5 for detection
        self.DataSets['LaneOriPointL']=510#360##420#480
        self.DataSets['LaneEndPointR']=770#1850#1100#1850
        self.DataSets['frm2start']=10#700#90#1600#900#90
        self.DataSets['laneoffset']=30#90#150#30#50
        self.DataSets['lanebandwidth']=30  
        self.DataSets['frame_height']=240#700#300#450# 300 for detection 
        self.DataSets['NumofLanes']=4 
        
        
        ###############################################
        # with 85 =pitch, and long covering FOV...annce to ground, very messy, and not good challenge
        self.DataSets['pitch']= '_0511_720q_'+str(85)
        self.DataSets['path']="/media/dom/TOSHIBA EXT/data12112020/720q85_noPeg2/goVL720q.avi"       
        self.DataSets['path_dpt']="/media/dom/TOSHIBA EXT/data12112020/720q85_noPeg2/goVD720q.avi"       
        self.DataSets['row0']=420# 5 for detection
        self.DataSets['LaneOriPointL']=480#360##420#480
        self.DataSets['LaneEndPointR']=800#1850#1100#1850
        self.DataSets['frm2start']=10#700#90#1600#900#90
        self.DataSets['laneoffset']=25#90#150#30#50
        self.DataSets['lanebandwidth']=35  
        self.DataSets['frame_height']=240#700#300#450# 300 for detection 
        self.DataSets['NumofLanes']=5         
        """

        """
        #########for statistic nalysis ################################
        # looks OK at beging.  by far
        # with 45=pitch, and long covering FOV...annce to ground, ...
        self.DataSets['pitch']= '_0511_720q_'+str(45)
        self.DataSets['path']="/media/dom/TOSHIBA EXT/data12112020/720q45_noPeg2_c3/goVL720q.avi"       
        self.DataSets['path_dpt']="/media/dom/TOSHIBA EXT/data12112020/720q45_noPeg2_c3/goVD720q.avi"       
        self.DataSets['row0']=430# 5 for detection
        self.DataSets['LaneOriPointL']=500#360##420#480
        self.DataSets['LaneEndPointR']=780#1850#1100#1850
        self.DataSets['frm2start']=10#700#90#1600#900#90
        self.DataSets['laneoffset']=30#90#150#30#50
        self.DataSets['lanebandwidth']=25  
        self.DataSets['frame_height']=240#700#300#450# 300 for detection 
        self.DataSets['NumofLanes']=5 
        """ 
        """
        ###########for statistic nalysis############################
        # good with 60=pitch, and long covering FOV...annce to ground, good...
        self.DataSets['pitch']= 'C21211P'+str(60)
        self.DataSets['path']="/media/dom/TOSHIBA EXT/data12112020/720q60_noPeg1_c2/goVL720q.avi"       
        self.DataSets['path_dpt']="/media/dom/TOSHIBA EXT/data12112020/720q60_noPeg1_c2/goVD720q.avi"       
        self.DataSets['row0']=420# 5 for detection
        self.DataSets['LaneOriPointL']=480#360##420#480
        self.DataSets['LaneEndPointR']=800#1850#1100#1850
        self.DataSets['frm2start']=10#700#90#1600#900#90
        self.DataSets['laneoffset']=35#90#150#30#50
        self.DataSets['lanebandwidth']=35  
        self.DataSets['frame_height']=240#700#300#450# 300 for detection 
        self.DataSets['NumofLanes']=5  
        """
        """        
        ###############for statistic nalysis#################
        # good with 60=pitch, and long covering FOV...annce to ground, good... low height
        self.DataSets['pitch']= '_0511_720q_'+str(60)
        self.DataSets['path']="/media/dom/TOSHIBA EXT/data12112020/720q60_noPeg2_c1/goVL720q.avi"       
        self.DataSets['path_dpt']="/media/dom/TOSHIBA EXT/data12112020/720q60_noPeg2_c1/goVD720q.avi"       
        self.DataSets['row0']=420# 5 for detection
        self.DataSets['LaneOriPointL']=510#360##420#480
        self.DataSets['LaneEndPointR']=770#1850#1100#1850
        self.DataSets['frm2start']=3#700#90#1600#900#90
        self.DataSets['laneoffset']=30#90#150#30#50
        self.DataSets['lanebandwidth']=35  
        self.DataSets['frame_height']=240#700#300#450# 300 for detection 
        self.DataSets['NumofLanes']=4 
        """ 
        ###############################################
        self.DataSets['low_green'] = np.array([20, 25,25])
        # self.DataSets['low_green'] = np.array([42, 0, 0])
        self.DataSets['high_green'] = np.array([100, 255, 255])

        # self.DataSets['low_green'] = np.array([25, 52, 72])
        # self.DataSets['high_green'] = np.array([102, 255, 255])

        # self.DataSets['low_green'] = np.array([18, 94, 140])
        # self.DataSets['high_green'] = np.array([48, 255, 255])

        # self.DataSets['low_green'] = np.array([30, 10, 10])
        # self.DataSets['high_green'] = np.array([100, 255, 255])

        # self.DataSets['low_green']=np.array([36, 25, 25])
        # self.DataSets['high_green']=np.array([126, 225, 225])
        
        return self.DataSets
    #####################################################################
    ####################################################################
    def data_26112020_4lanes_all(self): 
        
        # with 60=pitch, and long covering FOV...    anticlock runing         
        self.DataSets['pitch']= '_2611_720p_'+str(60)
        self.DataSets['path']="/media/dom/Elements/data/data_collection/data26112020/720p60_all/goVL720p.avi"       
        self.DataSets['path_dpt']="/media/dom/Elements/data/data_collection/720p60_all/goVD720p.avi"       
        self.DataSets['row0']=480# 5 for detection
        self.DataSets['LaneOriPointL']=320#420
        self.DataSets['LaneEndPointR']=960#1850#1100#1850
        self.DataSets['frm2start']=35#  35 - 70 for pegs
        self.DataSets['laneoffset']=500#90#150#30#50
        self.DataSets['lanebandwidth']=80
        self.DataSets['frame_height']=240#700#300#450# 300 for detection 
        self.DataSets['NumofLanes']=5
        ############################################################
        """
        # with 60=pitch, and long covering FOV...            
        self.DataSets['pitch']= '_2611_720p_'+str(60)
        self.DataSets['path']="/media/dom/TOSHIBA EXT/data26112020/720p60_all/goVL720p.avi"       
        self.DataSets['path_dpt']="/media/dom/TOSHIBA EXT/data26112020/720p60_all/goVD720p.avi"       
        self.DataSets['row0']=480# 5 for detection
        self.DataSets['LaneOriPointL']=370#420
        self.DataSets['LaneEndPointR']=910#1850#1100#1850
        self.DataSets['frm2start']=75#700#90#1600#900#90
        self.DataSets['laneoffset']=70#90#150#30#50
        self.DataSets['lanebandwidth']=60
        self.DataSets['frame_height']=240#700#300#450# 300 for detection 
        self.DataSets['NumofLanes']=5
        
        # another video cirlce clockwise data set 3 in paper wring gap = 30 cm
        # with 75=pitch, and long covering FOV... paper wring : gap = (830-450)/(3-1)=190 = 
        self.DataSets['pitch']= '_2611_720q_'+str(75)
        self.DataSets['path']="/media/dom/TOSHIBA EXT/data26112020/720q75_low/goVL720q.avi"       
        self.DataSets['path_dpt']="/media/dom/TOSHIBA EXT/data26112020/720q75_low/goVD720q.avi"       
        self.DataSets['row0']=460# 5 for detection
        self.DataSets['LaneOriPointL']=450#420
        self.DataSets['LaneEndPointR']=830#1850#1100#1850
        self.DataSets['frm2start']=15#  35 - 70 for pegs
        self.DataSets['laneoffset']=50#90#150#30#50
        self.DataSets['lanebandwidth']=60
        self.DataSets['frame_height']=240#700#300#450# 300 for detection 
        self.DataSets['NumofLanes']=3
        ##############################################################        
        """
     

        ###############################################
        self.DataSets['low_green'] = np.array([20, 25,25])
        # self.DataSets['low_green'] = np.array([42, 0, 0])
        self.DataSets['high_green'] = np.array([100, 255, 255])

        # self.DataSets['low_green'] = np.array([25, 52, 72])
        # self.DataSets['high_green'] = np.array([102, 255, 255])

        # self.DataSets['low_green'] = np.array([18, 94, 140])
        # self.DataSets['high_green'] = np.array([48, 255, 255])

        # self.DataSets['low_green'] = np.array([30, 10, 10])
        # self.DataSets['high_green'] = np.array([100, 255, 255])

        # self.DataSets['low_green']=np.array([36, 25, 25])
        # self.DataSets['high_green']=np.array([126, 225, 225])
        
        return self.DataSets
    #####################################################################

    def data_26112020_svo_all(self): 
        
        # with 60=pitch, and long covering FOV...    anticlock runing         
        self.DataSets['pitch']= '_2611svo_720p_'+str(60)
        self.DataSets['path']="/media/dom/TOSHIBA EXT/data26112020/data_svo/HD720_SN20484174_14-15-26.svo"       
        self.DataSets['path_dpt']="/media/dom/TOSHIBA EXT/data26112020/data_svo/HD720_SN20484174_14-15-26.svo"

                
        self.DataSets['row0']=480# 5 for detection
        self.DataSets['LaneOriPointL']=370#420
        self.DataSets['LaneEndPointR']=910#1850#1100#1850
        self.DataSets['frm2start']=35#  35 - 70 for pegs
        self.DataSets['laneoffset']=70#90#150#30#50
        self.DataSets['lanebandwidth']=60
        self.DataSets['frame_height']=240#700#300#450# 300 for detection 
        self.DataSets['NumofLanes']=5
        """
        ############################################################
        """
        # with 60=pitch, and long covering FOV...            
        self.DataSets['pitch']= '_2611_720p_'+str(60)
        self.DataSets['path']="/media/dom/TOSHIBA EXT/data26112020/data_svo/HD720_SN20484174_14-13-15.svo"       
        self.DataSets['path_dpt']="/media/dom/TOSHIBA EXT/data26112020/data_svo/HD720_SN20484174_14-13-15.svo"       
      
        self.DataSets['row0']=480# 5 for detection
        self.DataSets['LaneOriPointL']=370#420
        self.DataSets['LaneEndPointR']=910#1850#1100#1850
        self.DataSets['frm2start']=75#700#90#1600#900#90
        self.DataSets['laneoffset']=70#90#150#30#50
        self.DataSets['lanebandwidth']=60
        self.DataSets['frame_height']=240#700#300#450# 300 for detection 
        self.DataSets['NumofLanes']=5


        # with 75=pitch, and long covering FOV... not good 
        self.DataSets['pitch']= '_2611svo_720q_'+str(75)
        self.DataSets['path']="/media/dom/TOSHIBA EXT/data26112020/data_svo/HD720_SN20484174_14-13-15.svo"       
        self.DataSets['path_dpt']="/media/dom/TOSHIBA EXT/data26112020/data_svo/HD720_SN20484174_14-13-15.svo"       
      
        self.DataSets['row0']=460# 5 for detection
        self.DataSets['LaneOriPointL']=480#420
        self.DataSets['LaneEndPointR']=800#1850#1100#1850
        self.DataSets['frm2start']=15#  35 - 70 for pegs
        self.DataSets['laneoffset']=50#90#150#30#50
        self.DataSets['lanebandwidth']=50
        self.DataSets['frame_height']=240#700#300#450# 300 for detection 
        self.DataSets['NumofLanes']=3
        ##############################################################        
        
     

        ###############################################
        self.DataSets['low_green'] = np.array([20, 25,25])
        # self.DataSets['low_green'] = np.array([42, 0, 0])
        self.DataSets['high_green'] = np.array([100, 255, 255])

        # self.DataSets['low_green'] = np.array([25, 52, 72])
        # self.DataSets['high_green'] = np.array([102, 255, 255])

        # self.DataSets['low_green'] = np.array([18, 94, 140])
        # self.DataSets['high_green'] = np.array([48, 255, 255])

        # self.DataSets['low_green'] = np.array([30, 10, 10])
        # self.DataSets['high_green'] = np.array([100, 255, 255])

        # self.DataSets['low_green']=np.array([36, 25, 25])
        # self.DataSets['high_green']=np.array([126, 225, 225])
        
        return self.DataSets
    #####################################################################
    def data_10122020_svo_all(self): 
        
        # with 60=pitch, and long covering FOV...   bending lane, good for demon on lane bending following, it seems working!      
        self.DataSets['pitch']= '_1012svo_'+str(45)
        self.DataSets['path']="/media/dom/TOSHIBA EXT/data10122020/HD720_2.svo"       
        self.DataSets['path_dpt']="/media/dom/TOSHIBA EXT/data10122020/HD720_2.svo"

                
        self.DataSets['row0']=480# 5 for detection
        self.DataSets['LaneOriPointL']=520#420
        self.DataSets['LaneEndPointR']=760#1850#1100#1850
        self.DataSets['frm2start']=5#  35 - 70 for pegs
        self.DataSets['laneoffset']=45#90#150#30#50
        self.DataSets['lanebandwidth']=55
        self.DataSets['frame_height']=240#700#300#450# 300 for detection 
        self.DataSets['NumofLanes']=3
        """
        ############################################################
        """
        
        # with 60=pitch, and long covering ..   3550 frames         
        self.DataSets['pitch']= '_1012svo_3_'+str(45)
        self.DataSets['path']="/media/dom/TOSHIBA EXT/data10122020/HD720_3.svo"       
        self.DataSets['path_dpt']="/media/dom/TOSHIBA EXT/data10122020/HD720_3.svo"       
      
        self.DataSets['row0']=480# 5 for detection
        self.DataSets['LaneOriPointL']=520#420
        self.DataSets['LaneEndPointR']=760#1850#1100#1850
        self.DataSets['frm2start']=5#700#90#1600#900#90
        self.DataSets['laneoffset']=45#90#150#30#50
        self.DataSets['lanebandwidth']=55
        self.DataSets['frame_height']=240#700#300#450# 300 for detection 
        self.DataSets['NumofLanes']=3
                
        # data set 4 in paper gap = 15 cm
        # with 45=pitch, and long covering FOV... 15216 frames - 550m distance 
        # for paper writing 500 meters = 14600 frames gap = (760-520)/(3-1)=120 = 15 cm
        self.DataSets['pitch']= '1012svo_4_'+str(45)
        self.DataSets['path']="/media/dom/TOSHIBA EXT/data10122020/HD720_4.svo"       
        self.DataSets['path_dpt']="/media/dom/TOSHIBA EXT/data10122020/HD720_4.svo"       
      
        self.DataSets['row0']=480# 5 for detection
        self.DataSets['LaneOriPointL']=520#420
        self.DataSets['LaneEndPointR']=760#1850#1100#1850
        self.DataSets['frm2start']=1250#  35 - 70 for pegs
        self.DataSets['laneoffset']=55#90#150#30#50
        self.DataSets['lanebandwidth']=55
        self.DataSets['frame_height']=240#700#300#450# 300 for detection 
        self.DataSets['NumofLanes']=3
        ##############################################################      
        # not good , only for cropping image with 45=pitch, and long covering FOV... 15216 frames - 550m distance
        self.DataSets['pitch']= '1012svo_1_'+str(45)
        self.DataSets['path']="/media/dom/TOSHIBA EXT/data10122020/HD720_1.svo"       
        self.DataSets['path_dpt']="/media/dom/TOSHIBA EXT/data10122020/HD720_1.svo"       
      
        self.DataSets['row0']=480# 5 for detection
        self.DataSets['LaneOriPointL']=500#420
        self.DataSets['LaneEndPointR']=780#1850#1100#1850
        self.DataSets['frm2start']=60#  35 - 70 for pegs
        self.DataSets['laneoffset']=55#90#150#30#50
        self.DataSets['lanebandwidth']=55
        self.DataSets['frame_height']=240#700#300#450# 300 for detection 
        self.DataSets['NumofLanes']=3
        

        ###############################################
        self.DataSets['low_green'] = np.array([20, 25,25])
        # self.DataSets['low_green'] = np.array([42, 0, 0])
        self.DataSets['high_green'] = np.array([100, 255, 255])

        # self.DataSets['low_green'] = np.array([25, 52, 72])
        # self.DataSets['high_green'] = np.array([102, 255, 255])

        # self.DataSets['low_green'] = np.array([18, 94, 140])
        # self.DataSets['high_green'] = np.array([48, 255, 255])

        # self.DataSets['low_green'] = np.array([30, 10, 10])
        # self.DataSets['high_green'] = np.array([100, 255, 255])

        # self.DataSets['low_green']=np.array([36, 25, 25])
        # self.DataSets['high_green']=np.array([126, 225, 225])
        
        return self.DataSets
    #####################################################################

    #####################################################################
    def data_17122020_svo_all(self): 

        # overall this data set the angle pitch too small , >60
        
        self.DataSets['pitch']= '1712svo_6a_'+str(40)
        self.DataSets['path']="/media/dom/TOSHIBA EXT/data17122020/HD720_3a.svo"       
        self.DataSets['path_dpt']="/media/dom/TOSHIBA EXT/data17122020/HD720_3a.svo"      
      
        self.DataSets['row0']=480# 5 for detection
        self.DataSets['LaneOriPointL']=580#420
        self.DataSets['LaneEndPointR']=700#1850#1100#1850
        self.DataSets['frm2start']=100#  35 - 70 for pegs
        self.DataSets['laneoffset']=30#90#150#30#50
        self.DataSets['lanebandwidth']=25
        self.DataSets['frame_height']=240#700#300#450# 300 for detection 
        self.DataSets['NumofLanes']=3
        """


        # with 40=pitch, and long covering ..   100 frames   , it is darker a bit      

        self.DataSets['pitch']= '1712svo_6a_'+str(40)
        self.DataSets['path']="/media/dom/TOSHIBA EXT/data17122020/HD720_4a.svo"       
        self.DataSets['path_dpt']="/media/dom/TOSHIBA EXT/data17122020/HD720_4a.svo"      
      
        self.DataSets['row0']=480# 5 for detection
        self.DataSets['LaneOriPointL']=580#420
        self.DataSets['LaneEndPointR']=700#1850#1100#1850
        self.DataSets['frm2start']=100#  35 - 70 for pegs
        self.DataSets['laneoffset']=30#90#150#30#50
        self.DataSets['lanebandwidth']=25
        self.DataSets['frame_height']=240#700#300#450# 300 for detection 
        self.DataSets['NumofLanes']=3
        """
        """
        ############################################################
        """
        """
        # with 30=pitch, and long covering ..   100 frames   , it is darker a bit      
        self.DataSets['pitch']= '_1012svo_3_'+str(45)
        self.DataSets['pitch']= '1712svo_6a_'+str(30)
        self.DataSets['path']="/media/dom/TOSHIBA EXT/data17122020/HD720_5a.svo"       
        self.DataSets['path_dpt']="/media/dom/TOSHIBA EXT/data17122020/HD720_5a.svo"      
      
        self.DataSets['row0']=480# 5 for detection
        self.DataSets['LaneOriPointL']=580#420
        self.DataSets['LaneEndPointR']=700#1850#1100#1850
        self.DataSets['frm2start']=10#  35 - 70 for pegs
        self.DataSets['laneoffset']=35#90#150#30#50
        self.DataSets['lanebandwidth']=25
        self.DataSets['frame_height']=240#700#300#450# 300 for detection 
        self.DataSets['NumofLanes']=3
        """      
        """
        # with 30=pitch, too small ....... - 100m distance, it is darker a bit later afternoon  - not good!!!!!!!!!!!!!!!
        # when the lane gap is small, must be the pith >45 , facing down is better!
        self.DataSets['pitch']= '1712svo_6a_'+str(30)
        self.DataSets['path']="/media/dom/TOSHIBA EXT/data17122020/HD720_6a.svo"       
        self.DataSets['path_dpt']="/media/dom/TOSHIBA EXT/data17122020/HD720_6a.svo"       
      
        self.DataSets['row0']=480# 5 for detection
        self.DataSets['LaneOriPointL']=560#420
        self.DataSets['LaneEndPointR']=720#1850#1100#1850
        self.DataSets['frm2start']=10#  35 - 70 for pegs
        self.DataSets['laneoffset']=35#90#150#30#50
        self.DataSets['lanebandwidth']=20
        self.DataSets['frame_height']=240#700#300#450# 300 for detection 
        self.DataSets['NumofLanes']=4
        ##############################################################      
        """
        

        ###############not good for bright #############
        self.DataSets['low_green'] = np.array([20, 25,25])
        # self.DataSets['low_green'] = np.array([42, 0, 0])
        self.DataSets['high_green'] = np.array([100, 255, 255])

        # self.DataSets['low_green'] = np.array([25, 52, 72])
        # self.DataSets['high_green'] = np.array([102, 255, 255])

        # self.DataSets['low_green'] = np.array([18, 94, 140])
        # self.DataSets['high_green'] = np.array([48, 255, 255])

        # self.DataSets['low_green'] = np.array([30, 10, 10])
        # self.DataSets['high_green'] = np.array([100, 255, 255])

        # self.DataSets['low_green']=np.array([36, 25, 25])
        # self.DataSets['high_green']=np.array([126, 225, 225])
        
        return self.DataSets
    #####################################################################
    def data_05012021_svo_all(self): 

            # overall this data set the angle pitch too small , >60
            
            self.DataSets['pitch']= '0501svo_'+str(60)
            self.DataSets['path']="/media/dom/TOSHIBA EXT/data05012021/HD720_13-22-53.svo"       
            self.DataSets['path_dpt']="/media/dom/TOSHIBA EXT/data05012021/HD720_13-22-53.svo"      
        
            self.DataSets['row0']=300# 5 for detection
            self.DataSets['LaneOriPointL']=540#420
            self.DataSets['LaneEndPointR']=740#1850#1100#1850
            self.DataSets['frm2start']=220#  35 - 70 for pegs
            self.DataSets['laneoffset']=40#90#150#30#50
            self.DataSets['lanebandwidth']=35
            self.DataSets['frame_height']=240#700#300#450# 300 for detection 
            self.DataSets['NumofLanes']=3
            """    
            """
            self.DataSets['pitch']= '0501svo_'+str(45)
            self.DataSets['path']="/media/dom/TOSHIBA EXT/data05012021/HD720_13-34-44.svo"       
            self.DataSets['path_dpt']="/media/dom/TOSHIBA EXT/data05012021/HD720_13-34-44.svo"      
        
            self.DataSets['row0']=400# 5 for detection
            self.DataSets['LaneOriPointL']=540#420
            self.DataSets['LaneEndPointR']=740#1850#1100#1850
            self.DataSets['frm2start']=120#  35 - 70 for pegs
            self.DataSets['laneoffset']=50#90#150#30#50
            self.DataSets['lanebandwidth']=40
            self.DataSets['frame_height']=240#700#300#450# 300 for detection 
            self.DataSets['NumofLanes']=3


            self.DataSets['pitch']= '0501svo_'+str(45)
            self.DataSets['path']="/media/dom/TOSHIBA EXT/data05012021/HD720_13-41-05.svo"       
            self.DataSets['path_dpt']="/media/dom/TOSHIBA EXT/data05012021/HD720_13-41-05.svo"      
        
            self.DataSets['row0']=400# 5 for detection
            self.DataSets['LaneOriPointL']=540#420
            self.DataSets['LaneEndPointR']=740#1850#1100#1850
            self.DataSets['frm2start']=320#  35 - 70 for pegs
            self.DataSets['laneoffset']=50#90#150#30#50
            self.DataSets['lanebandwidth']=40
            self.DataSets['frame_height']=240#700#300#450# 300 for detection 
            self.DataSets['NumofLanes']=3

            ###############not good for bright #############
            self.DataSets['low_green'] = np.array([20, 25,25])
            # self.DataSets['low_green'] = np.array([42, 0, 0])
            self.DataSets['high_green'] = np.array([100, 255, 255])

            # self.DataSets['low_green'] = np.array([25, 52, 72])
            # self.DataSets['high_green'] = np.array([102, 255, 255])

            # self.DataSets['low_green'] = np.array([18, 94, 140])
            # self.DataSets['high_green'] = np.array([48, 255, 255])

            # self.DataSets['low_green'] = np.array([30, 10, 10])
            # self.DataSets['high_green'] = np.array([100, 255, 255])

            # self.DataSets['low_green']=np.array([36, 25, 25])
            # self.DataSets['high_green']=np.array([126, 225, 225])
            
            return self.DataSets
        #####################################################################

        #####################################################################
    def data_data12012021_svo_all(self): 
            # this one is not good enough!!!!!!!!!!!!!!, camera bias mounted, sadly
            # This set is  time: after noon , still not too bad in general:  small pitch = long view = less number lanes to choose 
            # = big lane offset = big lanewidth : this one prove , the pitch biger is better = faceing down more easy meassurens
            self.DataSets['pitch']= '1201svo_'+str(35)  # facing down less
            self.DataSets['path']="/media/dom/TOSHIBA EXT/data12012021/HD720_SN20484174_15-25-43.svo"       
            self.DataSets['path_dpt']="/media/dom/TOSHIBA EXT/data12012021/HD720_SN20484174_15-25-43.svo"      
        
            self.DataSets['row0']=320# 5 for detection
            self.DataSets['LaneOriPointL']=560#420
            self.DataSets['LaneEndPointR']=720#1850#1100#1850
            self.DataSets['frm2start']=10#  35 - 70 for pegs
            self.DataSets['laneoffset']=45#90#150#30#50
            self.DataSets['lanebandwidth']=30
            self.DataSets['frame_height']=240#700q#300#450# 300 for detection 
            self.DataSets['NumofLanes']=3
            
            # This set is  time: after noon , still not too bad in general:  small pitch = long view = less number lanes to choose 
            # = big lane offset = big lanewidth  good for partcile filer demon
            self.DataSets['pitch']= '1201svo_'+str(35)  # facing down less
            self.DataSets['path']="/media/dom/TOSHIBA EXT/data12012021/HD720_SN20484174_15-19-12.svo"       
            self.DataSets['path_dpt']="/media/dom/TOSHIBA EXT/data12012021/HD720_SN20484174_15-19-12.svo"      
        
            self.DataSets['row0']=320# 5 for detection
            self.DataSets['LaneOriPointL']=560#420
            self.DataSets['LaneEndPointR']=720#1850#1100#1850
            self.DataSets['frm2start']=10#  35 - 70 for pegs
            self.DataSets['laneoffset']=40#90#150#30#50
            self.DataSets['lanebandwidth']=30
            self.DataSets['frame_height']=240#700q#300#450# 300 for detection 
            self.DataSets['NumofLanes']=3
            
            # This set is  time: after noon , still not too bad in general:  small pitch = long view = less number lanes to choose 
            # = big lane offset = big lanewidth also good for particle filer
            self.DataSets['pitch']= '1201svo_'+str(35)  # facing down less
            self.DataSets['path']="/media/dom/TOSHIBA EXT/data12012021/HD720_SN20484174_15-17-02.svo"       
            self.DataSets['path_dpt']="/media/dom/TOSHIBA EXT/data12012021/HD720_SN20484174_15-17-02.svo"      
        
            self.DataSets['row0']=320# 5 for detection
            self.DataSets['LaneOriPointL']=560#420
            self.DataSets['LaneEndPointR']=720#1850#1100#1850
            self.DataSets['frm2start']=10#  35 - 70 for pegs
            self.DataSets['laneoffset']=40#90#150#30#50
            self.DataSets['lanebandwidth']=30
            self.DataSets['frame_height']=240#700q#300#450# 300 for detection 
            self.DataSets['NumofLanes']=3
            
            # This set is  time: after noon , still not too bad , good for particle filter
            self.DataSets['pitch']= '1201svo_'+str(40)
            self.DataSets['path']="/media/dom/TOSHIBA EXT/data12012021/HD720_SN20484174_14-17-09.svo"       
            self.DataSets['path_dpt']="/media/dom/TOSHIBA EXT/data12012021/HD720_SN20484174_14-17-09.svo"      
        
            self.DataSets['row0']=300# 5 for detection
            self.DataSets['LaneOriPointL']=530#420
            self.DataSets['LaneEndPointR']=750#1850#1100#1850
            self.DataSets['frm2start']=10#  35 - 70 for pegs
            self.DataSets['laneoffset']=30#90#150#30#50
            self.DataSets['lanebandwidth']=30
            self.DataSets['frame_height']=240#700q#300#450# 300 for detection 
            self.DataSets['NumofLanes']=4

            ## 4 lane settings are  good for particle filter ,better than 3 cos for this image 
            # 4 have more symetric on 1 wheel driving 

            # This set is  time: after noon , still not too bad
            self.DataSets['pitch']= '1201svo_'+str(40)
            self.DataSets['path']="/media/dom/TOSHIBA EXT/data12012021/HD720_SN20484174_14-17-09.svo"       
            self.DataSets['path_dpt']="/media/dom/TOSHIBA EXT/data12012021/HD720_SN20484174_14-17-09.svo"      
        
            self.DataSets['row0']=300# 5 for detection
            self.DataSets['LaneOriPointL']=530#420
            self.DataSets['LaneEndPointR']=750#1850#1100#1850
            self.DataSets['frm2start']=10#  35 - 70 for pegs
            self.DataSets['laneoffset']=30#90#150#30#50
            self.DataSets['lanebandwidth']=30
            self.DataSets['frame_height']=240#700q#300#450# 300 for detection 
            self.DataSets['NumofLanes']=4

            
            # ********** This set is the most suitalbe one , not to bad to make sense******time: afternoon*************
            self.DataSets['pitch']= '1201svo_'+str(40)
            self.DataSets['path']="/media/dom/TOSHIBA EXT/data12012021/HD720_SN20484174_14-14-46.svo"       
            self.DataSets['path_dpt']="/media/dom/TOSHIBA EXT/data12012021/HD720_SN20484174_14-14-46.svo"      
        
            self.DataSets['row0']=300# 5 for detection
            self.DataSets['LaneOriPointL']=530#420
            self.DataSets['LaneEndPointR']=750#1850#1100#1850
            self.DataSets['frm2start']=10#  35 - 70 for pegs
            self.DataSets['laneoffset']=30#90#150#30#50
            self.DataSets['lanebandwidth']=30
            self.DataSets['frame_height']=240#700q#300#450# 300 for detection 
            self.DataSets['NumofLanes']=4
            """
            """

            
            self.DataSets['pitch']= '1201svo_'+str(50)
            self.DataSets['path']="/media/dom/TOSHIBA EXT/data12012021/HD720_SN20484174_14-12-24.svo"       
            self.DataSets['path_dpt']="/media/dom/TOSHIBA EXT/data12012021/HD720_SN20484174_14-12-24.svo"      
        
            self.DataSets['row0']=300# 5 for detection
            self.DataSets['LaneOriPointL']=530#420
            self.DataSets['LaneEndPointR']=750#1850#1100#1850
            self.DataSets['frm2start']=10#  35 - 70 for pegs
            self.DataSets['laneoffset']=45#90#150#30#50
            self.DataSets['lanebandwidth']=30
            self.DataSets['frame_height']=240#700q#300#450# 300 for detection 
            self.DataSets['NumofLanes']=4
            
            # shadow is good for particle method as well
            self.DataSets['pitch']= '1201svo_'+str(40)
            self.DataSets['path']="/media/dom/TOSHIBA EXT/data12012021/HD720_SN20484174_14-10-40.svo"       
            self.DataSets['path_dpt']="/media/dom/TOSHIBA EXT/data12012021/HD720_SN20484174_14-10-40.svo"      
        
            self.DataSets['row0']=300# 5 for detection
            self.DataSets['LaneOriPointL']=520#420
            self.DataSets['LaneEndPointR']=760#1850#1100#1850
            self.DataSets['frm2start']=10#  35 - 70 for pegs
            self.DataSets['laneoffset']=40#90#150#30#50
            self.DataSets['lanebandwidth']=30
            self.DataSets['frame_height']=240#700q#300#450# 300 for detection 
            self.DataSets['NumofLanes']=4
            
            # bright is good for partcle as well, can used for paper later in particle filter
            self.DataSets['pitch']= '1201svo_'+str(45)
            self.DataSets['path']="/media/dom/TOSHIBA EXT/data12012021/HD720_SN20484174_13-13-32.svo"       
            self.DataSets['path_dpt']="/media/dom/TOSHIBA EXT/data12012021/HD720_SN20484174_13-13-32.svo"      
        
            self.DataSets['row0']=300# 5 for detection
            self.DataSets['LaneOriPointL']=530#420
            self.DataSets['LaneEndPointR']=750#1850#1100#1850
            self.DataSets['frm2start']=10#  35 - 70 for pegs
            self.DataSets['laneoffset']=45#90#150#30#50
            self.DataSets['lanebandwidth']=30
            self.DataSets['frame_height']=240#700#300#450# 300 for detection 
            self.DataSets['NumofLanes']=4
            
            """
            self.DataSets['pitch']= '1201svo_'+str(45)
            self.DataSets['path']="/media/dom/TOSHIBA EXT/data12012021/HD720_SN20484174_13-10-15.svo"       
            self.DataSets['path_dpt']="/media/dom/TOSHIBA EXT/data12012021/HD720_SN20484174_13-10-15.svo"      
        
            self.DataSets['row0']=300# 5 for detection
            self.DataSets['LaneOriPointL']=560#420
            self.DataSets['LaneEndPointR']=720#1850#1100#1850
            self.DataSets['frm2start']=10#  35 - 70 for pegs
            self.DataSets['laneoffset']=50#90#150#30#50
            self.DataSets['lanebandwidth']=40
            self.DataSets['frame_height']=240#700#300#450# 300 for detection 
            self.DataSets['NumofLanes']=3
            """
            ###############not good for bright #############
            self.DataSets['low_green'] = np.array([20, 25,25])
            # self.DataSets['low_green'] = np.array([42, 0, 0])
            self.DataSets['high_green'] = np.array([100, 255, 255])

            # self.DataSets['low_green'] = np.array([25, 52, 72])
            # self.DataSets['high_green'] = np.array([102, 255, 255])

            # self.DataSets['low_green'] = np.array([18, 94, 140])
            # self.DataSets['high_green'] = np.array([48, 255, 255])

            # self.DataSets['low_green'] = np.array([30, 10, 10])
            # self.DataSets['high_green'] = np.array([100, 255, 255])

            # self.DataSets['low_green']=np.array([36, 25, 25])
            # self.DataSets['high_green']=np.array([126, 225, 225])
            
            return self.DataSets
        #####################################################################

    def data_data22012021_svo_all(self):            
                 
            """
            """
            # low , pitch < = 45  the cross over crops presented , not too clear on lane rows
            self.DataSets['pitch']= '2201svo_'+str(45)  
            self.DataSets['path']="/media/dom/TOSHIBA EXT/data22012021/HD720_SN20484174_13-55-07.svo"       
            self.DataSets['path_dpt']="/media/dom/TOSHIBA EXT/data22012021/HD720_SN20484174_13-55-07.svo"      
        
            self.DataSets['row0']=340# 5 for detection
            self.DataSets['LaneOriPointL']=480#
            self.DataSets['LaneEndPointR']=800#
            self.DataSets['frm2start']=10# 
            self.DataSets['laneoffset']=55#
            self.DataSets['lanebandwidth']=45
            self.DataSets['frame_height']=240#700#300#450# 300 for detection 
            self.DataSets['NumofLanes']=4


            # low , pitch < = 45  the wheel running on the crops..............can used for paper in good condtion by water puddle.......
            self.DataSets['pitch']= '2201svo_'+str(45)  
            self.DataSets['path']="/media/dom/TOSHIBA EXT/data22012021/HD720_SN20484174_14-19-41.svo"       
            self.DataSets['path_dpt']="/media/dom/TOSHIBA EXT/data22012021/HD720_SN20484174_14-19-41.svo"      
        
            self.DataSets['row0']=410# 5 for detection
            self.DataSets['LaneOriPointL']=520#
            self.DataSets['LaneEndPointR']=760#
            self.DataSets['frm2start']=10# 
            self.DataSets['laneoffset']=50#
            self.DataSets['lanebandwidth']=45
            self.DataSets['frame_height']=240#700#300#450# 300 for detection 
            self.DataSets['NumofLanes']=3

            """
            # low , pitch > = 45 shadow but may have gap between , can be used for paper
            # also prove the right angle and height, can make camera fit 3 lane symetricly
            self.DataSets['pitch']= '2201svo_'+str(45)  
            self.DataSets['path']="/media/dom/TOSHIBA EXT/data22012021/HD720_SN20484174_14-16-20.svo"       
            self.DataSets['path_dpt']="/media/dom/TOSHIBA EXT/data22012021/HD720_SN20484174_14-16-20.svo"      
        
            self.DataSets['row0']=420# 5 for detection
            self.DataSets['LaneOriPointL']=540#
            self.DataSets['LaneEndPointR']=740#
            self.DataSets['frm2start']=10# 
            self.DataSets['laneoffset']=40#
            self.DataSets['lanebandwidth']=40
            self.DataSets['frame_height']=240#700#300#450# 300 for detection 
            self.DataSets['NumofLanes']=3
            
            # changing weather
            self.DataSets['pitch']= '2201svo_'+str(45)  
            self.DataSets['path']="/media/dom/TOSHIBA EXT/data22012021/HD720_SN20484174_12-34-01.svo"       
            self.DataSets['path_dpt']="/media/dom/TOSHIBA EXT/data22012021/HD720_SN20484174_12-34-01.svo"      
        
            self.DataSets['row0']=320# 5 for detection
            self.DataSets['LaneOriPointL']=480#
            self.DataSets['LaneEndPointR']=800#
            self.DataSets['frm2start']=10# 
            self.DataSets['laneoffset']=60#
            self.DataSets['lanebandwidth']=40
            self.DataSets['frame_height']=240#700#300#450# 300 for detection 
            self.DataSets['NumofLanes']=4
            
            
            # low , pitch > = 60 shadow
            self.DataSets['pitch']= '2201svo_'+str(60)  
            self.DataSets['path']="/media/dom/TOSHIBA EXT/data22012021/HD720_SN20484174_12-48-04.svo"       
            self.DataSets['path_dpt']="/media/dom/TOSHIBA EXT/data22012021/HD720_SN20484174_12-48-04.svo"      
        
            self.DataSets['row0']=340# 5 for detection
            self.DataSets['LaneOriPointL']=480#
            self.DataSets['LaneEndPointR']=800#
            self.DataSets['frm2start']=10# 
            self.DataSets['laneoffset']=60#
            self.DataSets['lanebandwidth']=40
            self.DataSets['frame_height']=240#700#300#450# 300 for detection 
            self.DataSets['NumofLanes']=4
            """
            ###############not good for bright #############
            self.DataSets['low_green'] = np.array([20, 25,25])
            # self.DataSets['low_green'] = np.array([42, 0, 0])
            self.DataSets['high_green'] = np.array([100, 255, 255])

            # self.DataSets['low_green'] = np.array([25, 52, 72])
            # self.DataSets['high_green'] = np.array([102, 255, 255])

            # self.DataSets['low_green'] = np.array([18, 94, 140])
            # self.DataSets['high_green'] = np.array([48, 255, 255])

            # self.DataSets['low_green'] = np.array([30, 10, 10])
            # self.DataSets['high_green'] = np.array([100, 255, 255])

            # self.DataSets['low_green']=np.array([36, 25, 25])
            # self.DataSets['high_green']=np.array([126, 225, 225])
            
            return self.DataSets
        #####################################################################
    def data_data01022021svo_com(self): 
        """
        on 01/02/20121 , wiilagm farm, puddles, sticky, hard to push forward.............
        the hough transform setting from 45 - 135 , now 60 -120, there are plenty features now.

        The camera height is about 1.3 m, and angle 45- 60q
        """
        #########slightly beter in this low pitch angle with 4 lanes 
        self.DataSets['pitch']= '0102svo10_'+str(45)  
        self.DataSets['path']="/media/dom/Elements/data/data_collection/data01022021/HD720_10.svo"       
        self.DataSets['path_dpt']="/media/dom/Elements/data/data_collection/data01022021/HD720_10.svo"      
    
        self.DataSets['row0']=240# 5 for detection
        self.DataSets['LaneOriPointL']=565#
        self.DataSets['LaneEndPointR']=715#
        self.DataSets['frm2start']=10# 
        self.DataSets['laneoffset']=30#
        self.DataSets['lanebandwidth']=25
        self.DataSets['frame_height']=240#700#300#450# 300 for detection 
        self.DataSets['NumofLanes']=4

        """
        """
        ###############not good for bright #############
        self.DataSets['low_green'] = np.array([20, 25,25])
        # self.DataSets['low_green'] = np.array([42, 0, 0])
        self.DataSets['high_green'] = np.array([100, 255, 255])
        return self.DataSets

    def data07032021GeN1_com(self):
        """
        George's No.1 near Nick's farm, sunndy , not very strong , most area dry , little bit bummping , not very hard
            , not much strong, 30 cm in gap with double row stuch together as a pair 
        """   

        """
        impressive, can be a demonstration - paper data set 19
        anti-clock , circling whole field, longest distandce, 400-500meter, for reseach oonly
        pitch kifte up up a little bit, so it is*10 about <=40, shunny, shadow, bummping, row fading, broken, upq/down hill...............

        the part for heading north  is with shadow, but still OK, can cropped for papers , frame 100 - 3650 for paper 
        
        no big bummpying, only you how to control the robot, slight shadow
        No.9  but used as dataset 19 in paper from   one with 11019 frames/513.8 MB, height = 1.4m, andgle < =40, heading east at befining ,  up/down hill, 
        ,  30 cm gap = 106 pixles wiht HT 75 - 105 degree, for spars rows,  about 140 m long in distance
        no big errors , challenge , lansStarPo = 53  max: 69.56 (1293) meanerror:  14.21x30/106 = 4.02cm

        """
        self.DataSets['pitch']= '0703svo9_'+str(75)
        self.DataSets['path']="/media/dom/Elements/data/data_collection/data07032021GeN1/HD720_9.svo"       
        self.DataSets['path_dpt']="/media/dom/Elements/data/data_collection/data07032021GeN1/HD720_9.svo"      

        """        self.DataSets['row0']=360# 5 for detection
        self.DataSets['LaneOriPointL']=480#420
        self.DataSets['LaneEndPointR']=800#1850#1100#1850
        self.DataSets['frm2start']=100#  35 - 70 for pegs
        self.DataSets['laneoffset']=570#90#150#30#50
        self.DataSets['lanebandwidth']=60
        self.DataSets['frame_height']=240#700#300#450# 300 for detection 
        self.DataSets['NumofLanes']=4       
        """
        self.DataSets['row0']=360# 5 for detection
        self.DataSets['LaneOriPointL']=480#420
        self.DataSets['LaneEndPointR']=800#1850#1100#1850
        self.DataSets['frm2start']=100#  35 - 70 for pegs
        self.DataSets['laneoffset']=60#90#150#30#50
        self.DataSets['lanebandwidth']=60
        self.DataSets['frame_height']=240#700#300#450# 300 for detection 
        self.DataSets['NumofLanes']=4       

        ###############not good for bright #############
        self.DataSets['low_green'] = np.array([20, 25,25])
        # self.DataSets['low_green'] = np.array([42, 0, 0])
        self.DataSets['high_green'] = np.array([100, 255, 255])
            
        return self.DataSets

    def data_live_setting(self):
        """
        for the real farm setting, calibration on various environment
        """

       
        self.DataSets['pitch']= 'rltime_'+str(75)
        self.DataSets['path']=''#"/media/dom/Elements/data/data_collection/data07032021GeN1/HD720_9.svo"       
        self.DataSets['path_dpt']=''#"/media/dom/Elements/data/data_collection/data07032021GeN1/HD720_9.svo"      

        self.DataSets['row0']=360# 5 for detection
        self.DataSets['LaneOriPointL']=480#420
        self.DataSets['LaneEndPointR']=800#1850#1100#1850
        self.DataSets['frm2start']=5#  35 - 70 for pegs
        self.DataSets['laneoffset']=60#90#150#30#50
        self.DataSets['lanebandwidth']=60
        self.DataSets['frame_height']=240#700#300#450# 300 for detection 
        self.DataSets['NumofLanes']=4       


        ###############not good for bright #############
        self.DataSets['low_green'] = np.array([20, 25,25])
        # self.DataSets['low_green'] = np.array([42, 0, 0])
        self.DataSets['high_green'] = np.array([100, 255, 255])
            
        return self.DataSets

    def data_data01022021svo_all(self):            
                 
            """
            on 01/02/20121 , wiilagm farm, puddles, sticky, hard to push forward.............
            the hough transform setting from 45 - 135 , now 60 -120, there are plenty features now.

            The camera height is about 1.3 m, and angle 45- 60q
            """
            # low , pitch < = 45  the cross over crops presented , not too clear on lane rows
            self.DataSets['pitch']= '0102svo_'+str(60)  
            self.DataSets['path']="/media/dom/TOSHIBA EXT/data01022021/HD720_SN20484174_13-46-41.svo"       
            self.DataSets['path_dpt']="/media/dom/TOSHIBA EXT/data01022021/HD720_SN20484174_13-46-41.svo"      
        
            self.DataSets['row0']=180# 5 for detection
            self.DataSets['LaneOriPointL']=585#
            self.DataSets['LaneEndPointR']=695#
            self.DataSets['frm2start']=10# 
            self.DataSets['laneoffset']=20#
            self.DataSets['lanebandwidth']=20
            self.DataSets['frame_height']=240#700#300#450# 300 for detection 
            self.DataSets['NumofLanes']=3

            # not too bad , and better than above
            self.DataSets['pitch']= '0102svo_'+str(55)  
            self.DataSets['path']="/media/dom/TOSHIBA EXT/data01022021/HD720_SN20484174_13-50-45.svo"       
            self.DataSets['path_dpt']="/media/dom/TOSHIBA EXT/data01022021/HD720_SN20484174_13-50-45.svo"      
        
            self.DataSets['row0']=230# 5 for detection
            self.DataSets['LaneOriPointL']=580#
            self.DataSets['LaneEndPointR']=700#
            self.DataSets['frm2start']=10# 
            self.DataSets['laneoffset']=20#
            self.DataSets['lanebandwidth']=25
            self.DataSets['frame_height']=240#700#300#450# 300 for detection 
            self.DataSets['NumofLanes']=3
            
            # crossover lands , middle of day
            self.DataSets['pitch']= '0102svo_'+str(55)  
            self.DataSets['path']="/media/dom/TOSHIBA EXT/data01022021/HD720_SN20484174_13-57-40.svo"       
            self.DataSets['path_dpt']="/media/dom/TOSHIBA EXT/data01022021/HD720_SN20484174_13-57-40.svo"      
        
            self.DataSets['row0']=230# 5 for detection
            self.DataSets['LaneOriPointL']=555#
            self.DataSets['LaneEndPointR']=725#
            self.DataSets['frm2start']=10# 
            self.DataSets['laneoffset']=25#
            self.DataSets['lanebandwidth']=25
            self.DataSets['frame_height']=240#700#300#450# 300 for detection 
            self.DataSets['NumofLanes']=4
            # not too bad
            self.DataSets['pitch']= '0102svo_'+str(55)  
            self.DataSets['path']="/media/dom/TOSHIBA EXT/data01022021/HD720_SN20484174_14-11-16.svo"       
            self.DataSets['path_dpt']="/media/dom/TOSHIBA EXT/data01022021/HD720_SN20484174_14-11-16.svo"      
        
            self.DataSets['row0']=230# 5 for detection
            self.DataSets['LaneOriPointL']=580#
            self.DataSets['LaneEndPointR']=700#
            self.DataSets['frm2start']=10# 
            self.DataSets['laneoffset']=30#
            self.DataSets['lanebandwidth']=25
            self.DataSets['frame_height']=240#700#300#450# 300 for detection 
            self.DataSets['NumofLanes']=3
            
            # after while is not too bad, e.g sencond half vedios
            self.DataSets['pitch']= '0102svo_'+str(55)  
            self.DataSets['path']="/media/dom/TOSHIBA EXT/data01022021/HD720_SN20484174_15-10-17.svo"       
            self.DataSets['path_dpt']="/media/dom/TOSHIBA EXT/data01022021/HD720_SN20484174_15-10-17.svo"      
        
            self.DataSets['row0']=230# 5 for detection
            self.DataSets['LaneOriPointL']=580#
            self.DataSets['LaneEndPointR']=700#
            self.DataSets['frm2start']=10# 
            self.DataSets['laneoffset']=30#
            self.DataSets['lanebandwidth']=25
            self.DataSets['frame_height']=240#700#300#450# 300 for detection 
            self.DataSets['NumofLanes']=3
            #####################average bad, not perfect
            self.DataSets['pitch']= '0102svo_'+str(55)  
            self.DataSets['path']="/media/dom/TOSHIBA EXT/data01022021/HD720_SN20484174_15-15-04.svo"       
            self.DataSets['path_dpt']="/media/dom/TOSHIBA EXT/data01022021/HD720_SN20484174_15-15-04.svo"      
        
            self.DataSets['row0']=230# 5 for detection
            self.DataSets['LaneOriPointL']=560#
            self.DataSets['LaneEndPointR']=720#
            self.DataSets['frm2start']=10# 
            self.DataSets['laneoffset']=25#
            self.DataSets['lanebandwidth']=25
            self.DataSets['frame_height']=240#700#300#450# 300 for detection 
            self.DataSets['NumofLanes']=4
            ####################################################
            self.DataSets['pitch']= '0102svo_'+str(45)  
            self.DataSets['path']="/media/dom/TOSHIBA EXT/data01022021/HD720_SN20484174_15-31-48.svo"       
            self.DataSets['path_dpt']="/media/dom/TOSHIBA EXT/data01022021/HD720_SN20484174_15-31-48.svo"      
        
            self.DataSets['row0']=240# 5 for detection
            self.DataSets['LaneOriPointL']=565#
            self.DataSets['LaneEndPointR']=715#
            self.DataSets['frm2start']=10# 
            self.DataSets['laneoffset']=32#
            self.DataSets['lanebandwidth']=25
            self.DataSets['frame_height']=240#700#300#450# 300 for detection 
            self.DataSets['NumofLanes']=4

            ####################################################
            self.DataSets['pitch']= '0102svo_'+str(45)  
            self.DataSets['path']="/media/dom/TOSHIBA EXT/data01022021/HD720_SN20484174_15-35-40.svo"       
            self.DataSets['path_dpt']="/media/dom/TOSHIBA EXT/data01022021/HD720_SN20484174_15-35-40.svo"      
        
            self.DataSets['row0']=240# 5 for detection
            self.DataSets['LaneOriPointL']=570#
            self.DataSets['LaneEndPointR']=710#
            self.DataSets['frm2start']=10# 
            self.DataSets['laneoffset']=30#
            self.DataSets['lanebandwidth']=25
            self.DataSets['frame_height']=240#700#300#450# 300 for detection 
            self.DataSets['NumofLanes']=4

            ####################################################
            self.DataSets['pitch']= '0102svo_'+str(45)  
            self.DataSets['path']="/media/dom/TOSHIBA EXT/data01022021/HD720_SN20484174_15-40-10.svo"       
            self.DataSets['path_dpt']="/media/dom/TOSHIBA EXT/data01022021/HD720_SN20484174_15-40-10.svo"      
        
            self.DataSets['row0']=240# 5 for detection
            self.DataSets['LaneOriPointL']=565#
            self.DataSets['LaneEndPointR']=715#
            self.DataSets['frm2start']=10# 
            self.DataSets['laneoffset']=30#
            self.DataSets['lanebandwidth']=25
            self.DataSets['frame_height']=240#700#300#450# 300 for detection 
            self.DataSets['NumofLanes']=4

            #########slightly beter in this low pitch angle with 4 lanes 
            self.DataSets['pitch']= '0102svo_'+str(45)  
            self.DataSets['path']="/media/dom/TOSHIBA EXT/data01022021/HD720_SN20484174_15-50-50.svo"       
            self.DataSets['path_dpt']="/media/dom/TOSHIBA EXT/data01022021/HD720_SN20484174_15-50-50.svo"      
        
            self.DataSets['row0']=240# 5 for detection
            self.DataSets['LaneOriPointL']=565#
            self.DataSets['LaneEndPointR']=715#
            self.DataSets['frm2start']=10# 
            self.DataSets['laneoffset']=30#
            self.DataSets['lanebandwidth']=25
            self.DataSets['frame_height']=240#700#300#450# 300 for detection 
            self.DataSets['NumofLanes']=4

            """
            """
            ###############not good for bright #############
            self.DataSets['low_green'] = np.array([20, 25,25])
            # self.DataSets['low_green'] = np.array([42, 0, 0])
            self.DataSets['high_green'] = np.array([100, 255, 255])

            # self.DataSets['low_green'] = np.array([25, 52, 72])
            # self.DataSets['high_green'] = np.array([102, 255, 255])

            # self.DataSets['low_green'] = np.array([18, 94, 140])
            # self.DataSets['high_green'] = np.array([48, 255, 255])

            # self.DataSets['low_green'] = np.array([30, 10, 10])
            # self.DataSets['high_green'] = np.array([100, 255, 255])

            # self.DataSets['low_green']=np.array([36, 25, 25])
            # self.DataSets['high_green']=np.array([126, 225, 225])
            
            return self.DataSets
        #####################################################################
class CCameraParamsSetting(object):
    def __init__(self,zed,sl,pth_ocv, pth_dpth,outputpth):
        self.video_ocv = pth_ocv#"/home/chfox/ARWAC/Essa-0/20190206_143625.mp4"
        self.video_dpt = pth_dpth#"/home/chfox/ARWAC/Essa-0/20190206_143625.mp4"
        self.grasslanes = {}
        self.firstFrame = True
        self.frame_num = 0
        self.frame_width=1060#int(camera.get(cv2.CAP_PROP_FRAME_WIDTH)*0.25)#640
        self.frame_height=1000#int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT)*0.25)#480
        self.camera = None
        self.camera_l = None
        self.camera_d = None
        self.totalNum_frame = inf #int(camera.get(cv2.CAP_PROP_FRAME_COUNT))
        self.video_image = np.zeros((self.frame_height,self.frame_width,3), np.uint8)
        self.video_counter = self.totalNum_frame*2
        
        self.fourcc1 = cv2.VideoWriter_fourcc(*'XVID')
        self.output_video_pth = outputpth#'/home/chfox/Documents/ARWAC/domReport/ref_images/Zedlivevideo.avi'
        # self.fourcc2 = cv2.VideoWriter_fourcc('M','J','P','G')
        self.out = cv2.VideoWriter(self.output_video_pth,self.fourcc1, 10, (self.frame_width,self.frame_height))
        # self.out2 = cv2.VideoWriter('/home/dom/Documents/domReport/ref_images/OriAvi.mp4',fourcc2, 10, (frame_width,frame_height))
        self.piplineSelet = 1 # The algorithm selection : simpfliled with easy field 1, complex 2 : RGB and Depth map 3 

class CRobtoParamSetting (object):
    def __init__(self,zed,sl):
        self.cam = zed
        self.sl = sl
        self.mode_depth = 0
        self.mode_point_cloud = 0
        self.count_save = 0
        self.depth_format = self.sl.MEASURE.DEPTH#XYZRGBA#DEPTH_FORMAT.DEPTH_FORMAT_PNG
        self.point_cloud_format = self.sl.MEASURE.XYZRGBA#POINT_CLOUD_FORMAT.POINT_CLOUD_FORMAT_XYZ_ASCII
        
        # Open the camera
        # Set runtime parameters after opening the camera
        self.runtime = self.sl.RuntimeParameters()
        # self.runtime.sensing_mode = self.sl.SENSING_MODE.SENSING_MODE_STANDARD
        self.runtime.sensing_mode = self.sl.SENSING_MODE.STANDARD

        # Prepare new image size to retrieve half-resolution images
        self.image_size = self.cam.get_camera_information().camera_resolution#get_resolution()
        self.new_width = int(self.image_size.width /2)
        self.new_height =int(self.image_size.height /2)
        self.image_size.width = int(self.image_size.width /2)
        self.image_size.height =int(self.image_size.height /2)

        # Declare your sl.Mat matrices
        self.image_zed = self.sl.Mat(self.new_width, self.new_height, self.sl.MAT_TYPE.U8_C4)#sl.MAT_TYPE.MAT_TYPE_8U_C4)  # this for display view
        self.depth_image_zed = self.sl.Mat(self.new_width, self.new_height, self.sl.MAT_TYPE.U8_C4)#MAT_TYPE.MAT_TYPE_8U_C4) # this for the display view
        # image_zed = sl.Mat(image_size.width, image_size.height, sl.MAT_TYPE.U8_C4)
        # depth_image_zed = sl.Mat(image_size.width, image_size.height, sl.MAT_TYPE.U8_C4)
        self.point_cloud = self.sl.Mat()

        self.key = ' '
        self.depth_zed = self.sl.Mat(self.new_width, self.new_height, self.sl.MAT_TYPE.F32_C3)#.MAT_TYPE_32F_C3)
        self.camera_pose = self.sl.Pose()
        self.viewer = ''#tv.PyTrackingViewer()
        self.py_translation = self.sl.Translation()


   