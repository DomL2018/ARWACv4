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

 #####################################################################
    


    def data_data01022021svo_all(self):            
                
        """
        on 01/02/20121 , wiilagm farm, puddles, sticky, hard to push forward.............
        the hough transform setting from 45 - 135 , now 60 -120, there are plenty features now.

        The camera height is about 1.3 m, and angle 45- 60q
        """
        # low , pitch < = 45  the cross over crops presented , not too clear on lane rows
        self.DataSets['pitch']= '0102svo_'+str(60)  
        self.DataSets['path']="/media/dom/8881-9CAC/data01022021/HD720_SN20484174_13-46-41.svo"       
        self.DataSets['path_dpt']="/media/dom/8881-9CAC/data01022021/HD720_SN20484174_13-46-41.svo"      
    
        self.DataSets['row0']=180# 5 for detection
        self.DataSets['LaneOriPointL']=585#
        self.DataSets['LaneEndPointR']=695#
        self.DataSets['frm2start']=10# 
        self.DataSets['laneoffset']=20#
        self.DataSets['lanebandwidth']=20
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


   