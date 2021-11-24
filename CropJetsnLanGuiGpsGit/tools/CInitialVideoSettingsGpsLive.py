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
        self.DataSets['pitch']= '3108svo_'+str(45)
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
        self.DataSets['disp_Height'] = 600# pixel for cropped display

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
        self.DataSets['pitch']= '3108svo_'+str(45)
        self.DataSets['path']="/media/dom/Elements/data/data_31082021_Hist/HD720_310821_1.svo"       
        self.DataSets['path_dpt']="/media/dom/Elements/data/data_31082021_Hist/HD720_310821_1.svo"

        ####################################    

        self.DataSets['row0']=150# 5 for detection
        self.DataSets['LaneOriPointL']=330#420
        self.DataSets['LaneEndPointR']=950#1850#1100#1850
        self.DataSets['frm2start']=10#  35 - 70 for pegs
        self.DataSets['laneoffset']=100#90#150#30#50
        self.DataSets['lanebandwidth']=60
        self.DataSets['frame_height']=480#700#300#450# 300 for detection 
        self.DataSets['NumofLanes']=5     
        self.DataSets['lanegap_cm'] = 30.0 # cm
        self.DataSets['disp_Height'] = 600# pixel for cropped display
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


   