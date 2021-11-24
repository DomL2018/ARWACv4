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
   
    def data31082021Hist_gps4Lanes(self): # 5 hoes
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
        self.DataSets['pitch']= '3108svo4L_'+str(45)
        self.DataSets['path']="/media/dom/Elements/data/data_31082021_Hist/HD720_310821_1.svo"       
        self.DataSets['path_dpt']="/media/dom/Elements/data/data_31082021_Hist/HD720_310821_1.svo"

        ####################################    

        self.DataSets['row0']=150# 5 for detection
        self.DataSets['LaneOriPointL']=440#420
        self.DataSets['LaneEndPointR']=840#1850#1100#1850
        self.DataSets['frm2start']=10#  35 - 70 for pegs
        self.DataSets['laneoffset']=65#90#150#30#50
        self.DataSets['lanebandwidth']=65
        self.DataSets['frame_height']=480#700#300#450# 300 for detection 
        self.DataSets['NumofLanes']=4
        self.DataSets['lanegap_cm'] = 30.0 # cm
        self.DataSets['disp_Height'] = 720# pixel for cropped display

        ###############not good for bright #############
        self.DataSets['low_green'] = np.array([20, 25,25])
        # self.DataSets['low_green'] = np.array([42, 0, 0])
        self.DataSets['high_green'] = np.array([100, 255, 255])
            
        return self.DataSets

    def data31082021Hist_gps5Lanes(self): # 6 hoes
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
        self.DataSets['pitch']= '3108svo5L_'+str(45)
        self.DataSets['path']="/media/dom/Elements/data/data_31082021_Hist/HD720_310821_1.svo"       
        self.DataSets['path_dpt']="/media/dom/Elements/data/data_31082021_Hist/HD720_310821_1.svo"

        ####################################    

        self.DataSets['row0']=150# 5 for detection
        self.DataSets['LaneOriPointL']=330#420
        self.DataSets['LaneEndPointR']=950#1850#1100#1850
        self.DataSets['frm2start']=160#  35 - 70 for pegs
        self.DataSets['laneoffset']=100#90#150#30#50
        self.DataSets['lanebandwidth']=85
        self.DataSets['frame_height']=480#700#300#450# 300 for detection 
        self.DataSets['NumofLanes']=5     
        self.DataSets['lanegap_cm'] = 30.0 # cm
        self.DataSets['disp_Height'] = 720# pixel for cropped display
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

        """      
        self.DataSets['row0']=360# 5 for detection
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
        self.DataSets['lanegap_cm'] = 30.0 # cm      
        self.DataSets['disp_Height'] = 600# pixel for cropped display
        ###############not good for bright #############
        self.DataSets['low_green'] = np.array([20, 25,25])
        # self.DataSets['low_green'] = np.array([42, 0, 0])
        self.DataSets['high_green'] = np.array([100, 255, 255])
            
        return self.DataSets
    #####################################################################
    #####################################################################
    def data28022021G2_2nd(self): 
        # George's farm , on 28/02/2021, fogy, light wind, slightly overlapping        
        #    in haconby village area
        """
        # 3rd better than last .2, gap: 123, 537.8MB, 11501 frames, 450 methers
        heiht about 1.4m, and pitch - 45  : big gap = 123, and too good to be for paper, 450 meters
        starpro = 46

        dataset 12 inside paper   mean error: 22.69  22.69x30/123 = 5.34 cm
        """
        
        self.DataSets['pitch']= '_2802G2_2nd_'+str(45)
        self.DataSets['path']="/media/dom/TOSHIBA EXT/data28022021G2/HD720_3.svo"       
        self.DataSets['path_dpt']="/media/dom/TOSHIBA EXT/data28022021G2/HD720_3.svo"
                
        self.DataSets['row0']=390# 5 for detection
        self.DataSets['LaneOriPointL']=455#420
        self.DataSets['LaneEndPointR']=825#1850#1100#1850
        self.DataSets['frm2start']=5#  35 - 70 for pegs
        self.DataSets['laneoffset']=65#90#150#30#50
        self.DataSets['lanebandwidth']=65
        self.DataSets['frame_height']=240#700#300#450# 300 for detection 
        self.DataSets['NumofLanes']=4
        self.DataSets['lanegap_cm'] = 30.0 # cm      
        self.DataSets['disp_Height'] = 640# pixel for cropped display
        ################################################################
        ################################################################
        """
        paper 6
        # 4th slightly south east facing , natural light shadow, HT : 60 - 120
        # a little bit challenge for views, shadow, bummpying, fading, heading east, good for paper , hieght = 1.4m
        # gap = 113, 3533 , distance : 240m  (actuall 400 meters,)
        # paper used   stratpo = 43  achiecd mean error 11.05 pixels = 11.05x30/113 = 2.93cm   error rate 9.78% accurate rate 90.02%
        self.DataSets['pitch']= '_2802svo_'+str(55)
        self.DataSets['path']="/media/dom/TOSHIBA EXT/data28022021G2/HD720_4.svo"       
        self.DataSets['path_dpt']="/media/dom/TOSHIBA EXT/data28022021G2/HD720_4.svo"
                
        self.DataSets['row0']=430# 5 for detection
        self.DataSets['LaneOriPointL']=470#420
        self.DataSets['LaneEndPointR']=810#1850#1100#1850
        self.DataSets['frm2start']=100#  35 - 70 for pegs
        self.DataSets['laneoffset']=65#90#150#30#50
        self.DataSets['lanebandwidth']=60
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
    def data01032021HaconbyDyke(self):  
        
        # two data set one for H\aconby, antoher william , narrow row

        # haconby black grass (30cm), 
        """
        low ,120 cm height, with about 60 in pitch, bumping , uneven hard surface,cloundy, all day
        """         
        # paper 8:
        # NO.2 good from haconby farm , with blackgrass,,,,may be for paper  in Haconby farm, with black grass there, and hard surface
        # 423.2MB/9084 340 meters, frames gap =135 with 30cm spacing, HT: 75 - 105, startpoint = 51, heading east, 30/135x11.25=2.5cm
        self.DataSets['pitch']= '0103byDyke3L'+str(55)
        self.DataSets['path']="/media/dom/TOSHIBA EXT/data01032021HaconbyDyke/HD720_2.svo"       
        self.DataSets['path_dpt']="/media/dom/TOSHIBA EXT/data01032021HaconbyDyke/HD720_2.svo"      
      
        self.DataSets['row0']=360# 5 for detection
        self.DataSets['LaneOriPointL']=505#420
        self.DataSets['LaneEndPointR']=775#1850#1100#1850
        self.DataSets['frm2start']=100#  35 - 70 for pegs
        self.DataSets['laneoffset']=55#90#150#30#50
        self.DataSets['lanebandwidth']=50
        self.DataSets['frame_height']=240#700#300#450# 300 for detection 
        self.DataSets['NumofLanes']=3
        self.DataSets['lanegap_cm'] = 30.0 # cm      
        self.DataSets['disp_Height'] = 600# pixel for cropped display
        """
               
        """
        """
        paper 7:
        ######################################################
        ## Willam data set NO.2, 12.5 cm - 15 cm in spacing bitch >=60, heading west, and clounding, no sunshine ,distance =140m
        ######################################################
        # Willam NO.2  height 1.3 m around, still hard, bumpping ,uneven, very density in spacing 12.5 -15 cm
        # hard to work on it, this one is not too bad, may be trying for paper, ......... HT: 75 - 105
        # gap = 63, 160.7MB/3430 frames  : StartPo = 24,   63x3 (189) = 12.5x2+15 (40cm) (4 rows in 3 gaps)  
        # mean error : 13.29 pixels = 2.81 cm (3.16cm for 15cm)
        self.DataSets['pitch']= '0103svo_'+str(60)
        self.DataSets['path']="/media/dom/TOSHIBA EXT/data01032021HaconbyDyke/william/HD720_2.svo"       
        self.DataSets['path_dpt']="/media/dom/TOSHIBA EXT/data01032021HaconbyDyke/william/HD720_2.svo"      
      
        self.DataSets['row0']=280# 260 for 240 height for detection
        self.DataSets['LaneOriPointL']=545#420
        self.DataSets['LaneEndPointR']=735#1850#1100#1850
        self.DataSets['frm2start']=100#  35 - 70 for pegs
        self.DataSets['laneoffset']=30#90#150#30#50
        self.DataSets['lanebandwidth']=30
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

    def data01032021HaconbyDyke_paperdata13(self): 
        # haconby black grass (30cm), 
        """
        low ,120 cm height, with about 60 in pitch, bumping , uneven hard surface,cloundy, all day
        """     
      
        """
        # NO.3  for paper data set 13:
        # pitch smaller now, and same height 1.3 m around, still hard, bumpping ,uneven,
        # can be paper use, good exaple for clustering
        # 461.0MB/9818 frames, gap = 126 lanesTrarPo = 48, with 11.76 pixels in mean error
        heading North, 360 meters long  11.76x30/126 = 2.8cm
        """
        self.DataSets['pitch']= '0103paperd13_'+str(45)
        self.DataSets['path']="/media/dom/TOSHIBA EXT/data01032021HaconbyDyke/HD720_3.svo"       
        self.DataSets['path_dpt']="/media/dom/TOSHIBA EXT/data01032021HaconbyDyke/HD720_3.svo"      
      
        self.DataSets['row0']=360# 5 for detection
        self.DataSets['LaneOriPointL']=450#420
        self.DataSets['LaneEndPointR']=830#1850#1100#1850
        self.DataSets['frm2start']=100#  35 - 70 for pegs
        self.DataSets['laneoffset']=75#90#150#30#50
        self.DataSets['lanebandwidth']=65
        self.DataSets['frame_height']=240#700#300#450# 300 for detection 
        self.DataSets['NumofLanes']=4    
        self.DataSets['lanegap_cm'] = 30.0 # cm      
        self.DataSets['disp_Height'] = 600# pixel for cropped display    
        
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

    def data01032021HaconbyDyke_paperdata14(self): 
        # haconby black grass (30cm), 
        """
        low ,120 cm height, with about 60 in pitch, bumping , uneven hard surface,cloundy, all day
        """        
                 
        # NO.4  for paper data set 14
        #  very bumpimg , unevent, and crop fading 
        # pitch smaller now, and same height 1.3 m around,
        # can be paper use , same as number 3, good exaple for clusteringq
        # gap = 136 pixles. 11231 frames/524.5MB pro = 51q meanerror = 16.85 pixels = 16.85x30/136 = 3.72 cm
        self.DataSets['pitch']= '0103svop14_'+str(45)
        self.DataSets['path']="/media/dom/TOSHIBA EXT/data01032021HaconbyDyke/HD720_4.svo"       
        self.DataSets['path_dpt']="/media/dom/TOSHIBA EXT/data01032021HaconbyDyke/HD720_4.svo"      
      
        self.DataSets['row0']=360# 5 for detection
        self.DataSets['LaneOriPointL']=435#420
        self.DataSets['LaneEndPointR']=845#1850#1100#1850
        self.DataSets['frm2start']=100#  35 - 70 for pegs
        self.DataSets['laneoffset']=80#90#150#30#50
        self.DataSets['lanebandwidth']=75
        self.DataSets['frame_height']=240#700#300#450# 300 for detection 
        self.DataSets['NumofLanes']=4             
        self.DataSets['lanegap_cm'] = 30.0 # cm      
        self.DataSets['disp_Height'] = 600# pixel for cropped display 
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
    def Data02032021Jo(self): 

        # overall this data set the angle pitch too small , >60
        # in Jonanthan's farm , with black grass , 15-18 cm spacing, cloundy, light wind, no sunshine
        # height 13cm, pitch 60, HT: 60 - 120       

        """
        short only 3874 frames    
        """

        """ 
        self.DataSets['pitch']= '0103svo_'+str(60)
        self.DataSets['path']="/media/dom/TOSHIBA EXT/Data02032021Jo/HD720_3.svo"       
        self.DataSets['path_dpt']="/media/dom/TOSHIBA EXT/Data02032021Jo/HD720_3.svo"      

        self.DataSets['row0']=300# 5 for detection
        self.DataSets['LaneOriPointL']=525#420
        self.DataSets['LaneEndPointR']=755#1850#1100#1850
        self.DataSets['frm2start']=120#  35 - 70 for pegs
        self.DataSets['laneoffset']=40#90#150#30#50
        self.DataSets['lanebandwidth']=35
        self.DataSets['frame_height']=240#700#300#450# 300 for detection 
        self.DataSets['NumofLanes']=4
        """

        """
        longest one with 6567 frames / 306.7MB  , may be for papers
        """
        """ self.DataSets['pitch']= '0103svo_'+str(60)
        self.DataSets['path']="/media/dom/TOSHIBA EXT/Data02032021Jo/HD720_4.svo"       
        self.DataSets['path_dpt']="/media/dom/TOSHIBA EXT/Data02032021Jo/HD720_4.svo"      

        self.DataSets['row0']=300# 5 for detection
        self.DataSets['LaneOriPointL']=540#420
        self.DataSets['LaneEndPointR']=740#1850#1100#1850
        self.DataSets['frm2start']=100#  35 - 70 for pegs
        self.DataSets['laneoffset']=40#90#150#30#50
        self.DataSets['lanebandwidth']=35
        self.DataSets['frame_height']=240#700#300#450# 300 for detection 
        self.DataSets['NumofLanes']=4"""

        """
        paper data set 9
        one with 3022 frames/140.9MB    , may be for papers
        gap: 66, 15-18 cm, starppos = 25, HT: 75 - 105, error: 9.42 oixles = 18/66x9.42 = 2.57 cm , heading northwest
        """
        self.DataSets['pitch']= '0203svoJo_'+str(55)
        self.DataSets['path']="/media/dom/TOSHIBA EXT/Data02032021Jo/HD720_5.svo"       
        self.DataSets['path_dpt']="/media/dom/TOSHIBA EXT/Data02032021Jo/HD720_5.svo"      

        self.DataSets['row0']=300# 5 for detection
        self.DataSets['LaneOriPointL']=535#420
        self.DataSets['LaneEndPointR']=745#1850#1100#1850
        self.DataSets['frm2start']=100#  35 - 70 for pegs
        self.DataSets['laneoffset']=40#90#150#30#50
        self.DataSets['lanebandwidth']=35
        self.DataSets['frame_height']=240#700#300#450# 300 for detection 
        self.DataSets['NumofLanes']=4    
        self.DataSets['lanegap_cm'] = 16.5 # cm      
        self.DataSets['disp_Height'] = 550# pixel for cropped display 

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

    def Data02032021Jo_p15(self): 

        # overall this data set the angle pitch too small , >60
        # in Jonanthan's farm , with black grass , 15-18 cm spacing, cloundy, light wind, no sunshine
        # height 13cm, pitch 60, HT: 60 - 120       

        """
        short only 3874 frames   -  paper data set 15  , here svo3
        one with 181.6MB    , may be for papers  (1870 used)
        gap: 80, 15-18 cm, starppos = 21, HT: 75 - 105, error: pixles = 15.36x18/80 = 3.456cm , heading southwest
        using 15 cm as gap:  15.36x15/80 = 28.8 which used in the paper dataset 15 as David's measuremen.
        max error in pixel :  40.50, and frame number :  1502
        """         
        self.DataSets['pitch']= '0203svoJoP15_'+str(60)
        self.DataSets['path']="/media/dom/TOSHIBA EXT/Data02032021Jo/HD720_3.svo"       
        self.DataSets['path_dpt']="/media/dom/TOSHIBA EXT/Data02032021Jo/HD720_3.svo"      

        self.DataSets['row0']=300# 5 for detection
        self.DataSets['LaneOriPointL']=520#420
        self.DataSets['LaneEndPointR']=760#1850#1100#1850
        self.DataSets['frm2start']=120#  35 - 70 for pegs
        self.DataSets['laneoffset']=45#90#150#30#50
        self.DataSets['lanebandwidth']=35
        self.DataSets['frame_height']=240#700#300#450# 300 for detection 
        self.DataSets['NumofLanes']=4    
        self.DataSets['lanegap_cm'] = 16.5 # cm      
        self.DataSets['disp_Height'] = 550# pixel for cropped display

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

    def Data02032021Jo_p16(self): 

        # overall this data set the angle pitch too small , >60
        # in Jonanthan's farm , with black grass , 15-18 cm spacing, cloundy, light wind, no sunshine
        # height 13cm, pitch 60, HT: 60 - 120    

        """
        paper : data set 16
        longest one with 6567 (2140 used )frames / 306.7MB  , may be for papers
        gap: 80, 15-18 cm, starppos = 21, HT: 75 - 105, error: pixles = 14.90x18/80 = 3.3525cm , heading northeast
        using 15 cm as gap:  14.90x15/80 = 27.9375mm which used in the paper dataset 16 as David's measuremen.
        max error in pixel :  44.87, and frame number :  1923
        """
        self.DataSets['pitch']= '0203svop16_'+str(60)
        self.DataSets['path']="/media/dom/TOSHIBA EXT/Data02032021Jo/HD720_4.svo"       
        self.DataSets['path_dpt']="/media/dom/TOSHIBA EXT/Data02032021Jo/HD720_4.svo"      

        self.DataSets['row0']=300# 5 for detection
        self.DataSets['LaneOriPointL']=520#420
        self.DataSets['LaneEndPointR']=760#1850#1100#1850
        self.DataSets['frm2start']=100#  35 - 70 for pegs
        self.DataSets['laneoffset']=45#90#150#30#50
        self.DataSets['lanebandwidth']=35
        self.DataSets['frame_height']=240#700#300#450# 300 for detection 
        self.DataSets['NumofLanes']=4
        self.DataSets['lanegap_cm'] = 15 # cm      
        self.DataSets['disp_Height'] = 550# pixel for cropped display

             

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
    def data05032021Stamf(self):
            """
            Stamford farm, cloundy, muddy, but not sticky too much, no much bummpying, 12.5-15 cm in gap,   two rows (12.5), again 15 cm between the pair 
            """      
                          
            
            """
            good for paper data set 10 , hahahsh not bad one......This one has been used now
            lower heihgt and angle, for density lane
            No.19  one with 1867 frames/84.7 MB,  height = 1.1m,slightly lower andgle > =45, heading east
            ,  12.5-15 cm gap = 60 pixles wiht HT 75 - 105 degree, for spars rows,  about 95 m long in distance
            Not bad , no big errors , still challenge , lansStarPo = 22, variance = 126.71
            This data set used up to 1675 frams for 90 meters: max error: 30.35 at fram 1029
            error 6.07 pixel = 6.07x12.5/60 = 1.265 cm
            """
            self.DataSets['pitch']= '0503svoStamf_'+str(45)
            self.DataSets['path']="/media/dom/TOSHIBA EXT/data05032021Stamf/HD720_19.svo"       
            self.DataSets['path_dpt']="/media/dom/TOSHIBA EXT/data05032021Stamf/HD720_19.svo"      

            self.DataSets['row0']=220# 5 for detection
            self.DataSets['LaneOriPointL']=550#420
            self.DataSets['LaneEndPointR']=730#1850#1100#1850
            self.DataSets['frm2start']=100#  35 - 70 for pegs
            self.DataSets['laneoffset']=30#90#150#30#50
            self.DataSets['lanebandwidth']=30
            self.DataSets['frame_height']=240#700#300#450# 300 for detection 
            self.DataSets['NumofLanes']=4
            self.DataSets['lanegap_cm'] = 12.5 # cm      
            self.DataSets['disp_Height'] = 460# pixel for cropped display

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


    def data05032021Stamf_p17(self):
            """
            Stamford farm, cloundy, muddy, but not sticky too much, no much bummpying, 12.5-15 cm in gap,   two rows (12.5), again 15 cm between the pair 
            """                  
        
            """
            good for papers.data set 17 .......... lower heihgt and angle, for density lane
            svo No.16  one with 1626 (1495)frames/75.8 MB,  height = 1.2m,slightly lower andgle > =65, heading Southwest
            ,  12.5-15 cm gap = 53 pixles wiht HT 75 - 105 degree, for spars rows,  about 110 m long in distance
            Not bad , no big errors , still challenge , lansStarPo = 25, max error = 30.06 (1351)
            meanerror: 6.79x12.5/53 = 1.60cm

            """
            self.DataSets['pitch']= '0503svop17_'+str(65)
            self.DataSets['path']="/media/dom/TOSHIBA EXT/data05032021Stamf/HD720_16.svo"       
            self.DataSets['path_dpt']="/media/dom/TOSHIBA EXT/data05032021Stamf/HD720_16.svo"      

            self.DataSets['row0']=300# 5 for detection
            self.DataSets['LaneOriPointL']=560#420
            self.DataSets['LaneEndPointR']=720#1850#1100#1850
            self.DataSets['frm2start']=100#  35 - 70 for pegs
            self.DataSets['laneoffset']=30#90#150#30#50
            self.DataSets['lanebandwidth']=30
            self.DataSets['frame_height']=240#700#300#450# 300 for detection 
            self.DataSets['NumofLanes']=4                 
            
            self.DataSets['lanegap_cm'] = 12.5 # cm      
            self.DataSets['disp_Height'] = 530# pixel for cropped display
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

    def data05032021Stamf_p18(self):
        """
        Stamford farm, cloundy, muddy, but not sticky too much, no much bummpying, 12.5-15 cm in gap,   two rows (12.5), again 15 cm between the pair 
        """      
            

        """
        2nd logest one, can be...........paper 18..
        No.11  one with 4017 frames/187.7 MB,  height = 1.3m, andgle > =45, heading east distance: 130m about
        ,  12.5-15 cm gap = 53 pixles wiht HT 75 - 105 degree, for spars rows,  about 150 m long in distance
        Not bad , no big errors , still challenge , lansStarPo = 20  : 8.50x12.5/53 = 2.0cm
        max: 32.47 at frame number 38

        """
        self.DataSets['pitch']= '0503svo11_'+str(70)
        self.DataSets['path']="/media/dom/TOSHIBA EXT/data05032021Stamf/HD720_11.svo"       
        self.DataSets['path_dpt']="/media/dom/TOSHIBA EXT/data05032021Stamf/HD720_11.svo"      

        self.DataSets['row0']=320# 5 for detection
        self.DataSets['LaneOriPointL']=560#420
        self.DataSets['LaneEndPointR']=720#1850#1100#1850
        self.DataSets['frm2start']=60#  35 - 70 for pegs
        self.DataSets['laneoffset']=35#90#150#30#50
        self.DataSets['lanebandwidth']=30
        self.DataSets['frame_height']=240#700#300#450# 300 for detection 
        self.DataSets['NumofLanes']=4      
        self.DataSets['lanegap_cm'] = 12.5 # cm      
        self.DataSets['disp_Height'] = 550# pixel for cropped display

        """
        paper........hahahaha, control very very well, the best..........
        No.13  one with 3225 frames/149.6 MB,  height = 1.3m, andgle > =45, heading west
        ,  12.5-15 cm gap = 53 pixles wiht HT 75 - 105 degree, for spars rows,  about 150 m long in distance
        Not bad , no big errors , still challenge , lansStarPo = 20

        
        self.DataSets['pitch']= '0503svo_'+str(65)
        self.DataSets['path']="/media/dom/TOSHIBA EXT/data05032021Stamf/HD720_13.svo"       
        self.DataSets['path_dpt']="/media/dom/TOSHIBA EXT/data05032021Stamf/HD720_13.svo"      

        self.DataSets['row0']=320# 5 for detection
        self.DataSets['LaneOriPointL']=560#420
        self.DataSets['LaneEndPointR']=720#1850#1100#1850
        self.DataSets['frm2start']=100#  35 - 70 for pegs
        self.DataSets['laneoffset']=25#90#150#30#50
        self.DataSets['lanebandwidth']=30
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
    def data07032021GeN1(self):
        """
        George's No.1 near Nick's farm, sunndy , not very strong , most area dry , little bit bummping , not very hard
            , not much strong, 30 cm in gap with double row stuch together as a pair 
        """          

        """
        good one , as sample for lower height , and big pitch (facing down!)  - paper data set 11
        lower height be with bit pitch angle >60 could be better in compatibility
        with big gap, the too lowe may not good as imageable, and more sensitive to the  position.......        
        No.13  one with 3126 frames/145.0 MB, height = 1.1m, with pitch down as andgle < =60, heading west at uphill, 
        ,  30 cm gap = 113 pixles wiht HT 75 - 105 degree, for spars rows,  about 80 m long in distance
        no big errors , challenge , lansStarPo = 39,  
        max erro: 62.20 pixels:  mean error: 13.06x30/113 = 3.467 cm, heading east, sunny, bumping
        paper dataset 11
        """
        self.DataSets['pitch']= '0703svoGeN1_'+str(60)
        self.DataSets['path']="/media/dom/TOSHIBA EXT/data07032021GeN1/HD720_13.svo"       
        self.DataSets['path_dpt']="/media/dom/TOSHIBA EXT/data07032021GeN1/HD720_13.svo"      

        self.DataSets['row0']=260# 5 for detection
        self.DataSets['LaneOriPointL']=470
        self.DataSets['LaneEndPointR']=810#1850#1100#1850
        self.DataSets['frm2start']=100#  35 - 70 for pegs
        self.DataSets['laneoffset']=55#90#150#30#50
        self.DataSets['lanebandwidth']=60
        self.DataSets['frame_height']=240#700#300#450# 300 for detection 
        self.DataSets['NumofLanes']=4
        self.DataSets['lanegap_cm'] = 30 # cm      
        self.DataSets['disp_Height'] = 510# pixel for cropped display

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


    def data07032021GeN1_p19(self):
        """
        George's No.1 near Nick's farm, sunndy , not very strong , most area dry , little bit bummping , not very hard
            , not much strong, 30 cm in gap with double row stuch together as a pair 
        """   

        """
        impressive, can be a demonstration - paper data set 19
        anti-clock , circling whole field, longest distandce, 400-500meter, for reseach oonly
        pitch kifte up up a little bit, so it is about <=40, shunny, shadow, bummping, row fading, broken, upq/down hill...............

        the part for heading north  is with shadow, but still OK, can cropped for papers , frame 100 - 3650 for paper 
        
        no big bummpying, only you how to control the robot, slight shadow
        No.9  but used as dataset 19 in paper from   one with 11019 frames/513.8 MB, height = 1.4m, andgle < =40, heading east at befining ,  up/down hill, 
        ,  30 cm gap = 106 pixles wiht HT 75 - 105 degree, for spars rows,  about 140 m long in distance
        no big errors , challenge , lansStarPo = 53  max: 69.56 (1293) meanerror:  14.21x30/106 = 4.02cm

        """
        self.DataSets['pitch']= '0703svoP19_'+str(75)
        self.DataSets['path']="/media/dom/TOSHIBA EXT/data07032021GeN1/HD720_9.svo"       
        self.DataSets['path_dpt']="/media/dom/TOSHIBA EXT/data07032021GeN1/HD720_9.svo"      

        self.DataSets['row0']=360# 5 for detection
        self.DataSets['LaneOriPointL']=480#420
        self.DataSets['LaneEndPointR']=800#1850#1100#1850
        self.DataSets['frm2start']=100#  35 - 70 for pegs
        self.DataSets['laneoffset']=60#90#150#30#50
        self.DataSets['lanebandwidth']=60
        self.DataSets['frame_height']=240#700#300#450# 300 for detection 
        self.DataSets['NumofLanes']=4
        self.DataSets['lanegap_cm'] = 30 # cm      
        self.DataSets['disp_Height'] = 610# pixel for cropped display


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

    def data07032021GeN1_p20(self):
        """
        George's No.1 near Nick's farm, sunndy , not very strong , most area dry , little bit bummping , not very hard
            , not much strong, 30 cm in gap with double row stuch together as a pair 
        """      

        """
        for paper,  haaha.....data set 20 in paper
        
        no big bummpying, only you how to control the robot, slight shadow
        No.8  svo one with 3618 frames/166.5 MB, height = 1.4m, andgle < =45, heading west ,  up hill, 
        ,  30 cm gap = 106 pixles wiht HT 75 - 105 degree, for spars rows,  about 80 m long in distance
         big error = 62.15 (1046) , less challenge , lansStarPo = 53,  meanerror: 13.29x30/106 = 3.761cm

        """
        self.DataSets['pitch']= '0703svoP20_'+str(45)
        self.DataSets['path']="/media/dom/TOSHIBA EXT/data07032021GeN1/HD720_8.svo"       
        self.DataSets['path_dpt']="/media/dom/TOSHIBA EXT/data07032021GeN1/HD720_8.svo"      

        self.DataSets['row0']=380# 5 for detection
        self.DataSets['LaneOriPointL']=480#420
        self.DataSets['LaneEndPointR']=800#1850#1100#1850
        self.DataSets['frm2start']=50#  35 - 70 for pegs
        self.DataSets['laneoffset']=55#90#150#30#50
        self.DataSets['lanebandwidth']=60
        self.DataSets['frame_height']=240#700#300#450# 300 for detection 
        self.DataSets['NumofLanes']=4      
        self.DataSets['lanegap_cm'] = 30 # cm      
        self.DataSets['disp_Height'] = 625# pixel for cropped display


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

    def data09052021AstroTurf_Hm(self):
        """
        lying on the garden floow for 
        """      

        """
        for paper,  haaha.....data set 20 in paper
        
        no big bummpying, only you how to control the robot, slight shadow
        No.8  svo one with 3618 frames/166.5 MB, height = 1.4m, andgle < =45, heading west ,  up hill, 
        ,  30 cm gap = 106 pixles wiht HT 75 - 105 degree, for spars rows,  about 80 m long in distance
         big error = 62.15 (1046) , less challenge , lansStarPo = 53,  meanerror: 13.29x30/106 = 3.761cm

        """
        self.DataSets['pitch']= '0905svo7_'+str(65)
        self.DataSets['path']="/media/dom/Elements/data/data_collection/Astra_home/HD720_7.svo"       
        self.DataSets['path_dpt']="/media/dom/Elements/data/data_collection/Astra_home/HD720_7.svo"      

        self.DataSets['row0']=400# 5 for detection
        self.DataSets['LaneOriPointL']=520#420
        self.DataSets['LaneEndPointR']=760#1850#1100#1850
        self.DataSets['frm2start']=25#  35 - 70 for pegs
        self.DataSets['laneoffset']=40#90#150#30#50
        self.DataSets['lanebandwidth']=45
        self.DataSets['frame_height']=240#700#300#450# 300 for detection 
        self.DataSets['NumofLanes']=4     
        


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
    def data_live_setting_0(self):
        """
        for the real farm setting, calibration on various environment
        """

       
        self.DataSets['pitch']= 'rltime_'+str(75)
        self.DataSets['path']=''#"/media/dom/Elements/data/data_collection/data07032021GeN1/HD720_9.svo"       
        self.DataSets['path_dpt']=''#"/media/dom/Elements/d#ata/data_collection/data07032021GeN1/HD720_9.svo"      

        self.DataSets['row0']=200#240# 5 for detection
        self.DataSets['LaneOriPointL']=420#560#420
        self.DataSets['LaneEndPointR']=860#720#870#1850#1100#1850
        self.DataSets['frm2start']=10#  35 - 70 for pegs
        self.DataSets['laneoffset']=80#90#150#30#50
        self.DataSets['lanebandwidth']=70
        self.DataSets['frame_height']=480#700#300#450# 300 for detection 
        self.DataSets['NumofLanes']=3     
        self.DataSets['lanegap_cm'] = 30.0 # cm
        self.DataSets['disp_Height'] = 600# pixel for cropped display
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

    def data_live_setting(self):
        """
        for the real farm setting, calibration on various environment
        """

       
        self.DataSets['pitch']= 'rltime_'+str(75)
        self.DataSets['path']=''#"/media/dom/Elements/data/data_collection/data07032021GeN1/HD720_9.svo"       
        self.DataSets['path_dpt']=''#"/media/dom/Elements/d#ata/data_collection/data07032021GeN1/HD720_9.svo"      

        self.DataSets['row0']=100# 5 for detection
        self.DataSets['LaneOriPointL']=485#420
        self.DataSets['LaneEndPointR']=795#1850#1100#1850
        self.DataSets['frm2start']=10#  35 - 70 for pegs
        self.DataSets['laneoffset']=90#90#150#30#50
        self.DataSets['lanebandwidth']=80
        self.DataSets['frame_height']=480#700#300#450# 300 for detection 
        self.DataSets['NumofLanes']=3    
        self.DataSets['lanegap_cm'] = 30.0 # cm
        self.DataSets['disp_Height'] = 600# pixel for cropped display
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


   