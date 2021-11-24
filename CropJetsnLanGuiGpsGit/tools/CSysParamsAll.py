import os
# from pickle import TRUE
# from re import S
import sys,time
import numpy as np
from numpy import inf
import pyzed.sl as sl
import cv2
# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.switch_backend('agg')
# from keras_yolo3.yolo import YOLO
# from tools_advanced.configuration import Configuration
# configuration = Configuration()
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

class CLanesParams(object):
    def __init__(self):
        
        # self.sl = None
        self.runtime = None #self.sl.RuntimeParameters()
        # self.runtime.sensing_mode = None #self.sl.SENSING_MODE.STANDARD
        self.Lane_GTstr = 'Current True Pos: ' # simulatie hose postion
        # self.runtime.sensing_mode = self.sl.SENSING_MODE.SENSING_MODE_STANDARD
        self.m_KfDnnDet = object()
        self.campitch='Laptop'  # setting on 'tidy ' image
        self.video_file = ''
        self.outRes = None
        self.Numoflanes=4 #total number of rows covered within ROI
        self.laneoffset =45
        self.isVidRecod = False
        self.recording_counter=0 
        self.PixelSize=1.0 # mm #configuration.config['laneparams']['PixelSize'] #mm        
        self.Ctrl_OfsetPos = {} # the abosolute position of detcted lanes during movements
        self.lanegap_cm = 30.00 # lane gap in physical world
        self.disp_Height = 720  # for displaying original image height
        # communication purpse, and measurement on the turning points
        self.ctrFlag = 1 # 1 and 5 #ctrl the selection of methods 1 -6
        self.dip_flag = False # control debugging the pic displaying
        self.sim_out = 0.0
        self.last_out = self.sim_out
        self.sleepT = 0.1# pause time for data com            
        self.rb_sleepT = 0.0
        self.avg_value=0.0
        self.msg_scl=0.0
        self.msg_vec = list()
        self.scale = -10.0 # control the normalization
        self.ctrFlag = 1 # selection of methods
        self.Lbase=200.0 #cm# leght of robot base
        self.Ibase = 240.0 # pixes : height detection windows size
        self.Wbase = self.Ibase*self.PixelSize # approximately , pixel is squared - detection windows height
        self.hose_offset = 0.0 # hose control offset
        ##################################################################
 
        self.LaneOriPoint = 0# for the lane intialization     
        
        self.LaneEndPoint=1880

        self.lanebandwith = 30
        self.lanegap=100
        self.laneStartPo = 20    #origin point for tracking 
        self.upd_thrsh = 10
        self.pf_start = 0

        self.row0 = 25
        self.col0 = 102
        self.fph = 0 # focus point location
        self.roi_height=0.8 # ratio heigth of ROI to focus point 
        self.phm = 24.0# meter

        self.lane_width=3.7  # meter
        self.lane_length=self.phm
        self.queue_size=32 # number of wins in quenue for ..

        self.svResPath = '/media/dom/Elements/data/data_processing/unet_filter/tst150721'
        self.w0 = 52
        self.h0 = 225
       
        ###########################
        self.HSVLowGreenClor = 42
        self.HSVHighGreenClor = 106

        self.HSVLowGreenSatur = 25
        self.HSVHighGreenSatur = 225

        self.HSVLowGreenVal = 25
        self.msg2ctr = 0.0 # data for navitation
        self.msg2ang = 0.0 # data for maninpulation


        """
        self.low_green = np.array([30, 0, 0])# for the purpose of putput info for dispimg
        self.high_green = np.array([100, 255, 255])# propective transform matrix    
        """
        self.frm2strart=75   # prospecitve transform
        self.avi_width=25   # prospecitve transform
        self.avi_height=25#25 #15  #25   
         
        ### depth map
        """self.FullDptRow0 = 0
        self.FullDptH0=375
        self.FullDptCol0 = 200
        self.FullDptW0=300
        self.dispimg_l='' # for the purpose of output processed results
        self.dispimg_r=''
        self.dispinfo =''# for the purpose of putput info for dispimg
        """
        self.minLineLength = 4 # Hough tranform the segments' minimum lengh : incrasing by number of lines decreasing
        self.maxLineGap = 2 # Hough transform , the segemtns's manximum gap increasing by number lines decreasing
     
        self.minLineLength_dpt = 50 # Hough tranform the segments' minimum lengh : incrasing by number of lines decreasing
        self.maxLineGap_dpt = 6 # Hough transform , the segemtns's manximum gap increasing by number lines decreasing
        ##########################################
        ##########################################      
        

        self.mt ='' # propective transform matrix  
        self.mt_inv = '' # prospective transform matix - inverse
        ########################################################
        self.Ctrl_Ofset={}# hold the output of offset for each indiviudal lane in metrics of mm
        self.Lane_GTs={}# hold the ground truth of each indiviudal lane in metrics of mm or pixles
        self.Lane_GTs_Prv={}
        self.Detec_Err = {}
        self.Lane_Ofset_Total=list()# hold the all offsets as whole so to do later analysis
        self.Lane_GTs_keys=list() # hold the lane key for the groudtruth records
        self.Rbt_Ofset=0.0
        self.mtricsize = 30 #cm the pysical size of lane gap : 12.,15,18, 30cm
        self.focal_point = [120,120]
        #########################################################
        self.pitch = 0
        self.path=''       
        # self.LaneOriPoint=0 # for the purpose of output processed results
        # self.LaneEndPoint=300
        # self.lanebandwith=80
        self.crped_h0 =0# for the purpose of putput info for 
        self.crped_w0 =400# for the purpose of putput info for dispimg
        self.crped_row0 = 240
        self.crped_col0 =400# for the purpose of putput info for dispimg   
        self.orig_frame = None
       
        
        """
        # Range for lower red
        self.low_red_lowrange = np.array([0,120,70])# 170 for the purpose of putput info for dispimg
        self.high_red_lowrange = np.array([10,255,255])# propective transform matrix  
        # Range for upper range
        self.low_red_uprange = np.array([170,120,70])# 170 for the purpose of putput info for dispimg
        self.high_red_uprange = np.array([179,255,255])# propective transform matrix      


        # Range for lower red - best
        self.low_red_lowrange = np.array([0,50,50])# 170 for the purpose of putput info for dispimg
        self.high_red_lowrange = np.array([10,255,255])# propective transform matrix  
        # Range for upper range
        self.low_red_uprange = np.array([110,50,50])# 170 for the purpose of putput info for dispimg
        self.high_red_uprange = np.array([130,255,255])# propective transform matrix      


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
        """
        self.low_green = np.array([20, 25,25])
        # self.DataSets['low_green'] = np.array([42, 0, 0])
        self.high_green = np.array([100, 255, 255])

        # class CAlgrthmParams(object):
        # def __init__(self,zed,sl,pth_ocv, pth_dpth,outputpth):
        self.video_ocv = ''#"/home/chfox/ARWAC/Essa-0/20190206_143625.mp4"
        self.video_dpt = ''#"/home/chfox/ARWAC/Essa-0/20190206_143625.mp4"
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
        self.ToIntializeSetting = False
        self.filter_start=0 # counting for in
        self.upd_thrsh=18 # updating started for tracking
        self.grasslanes={}
        self.issvo = False
        self.isavi = False
        self.m_detector = object()
        
        # self.fourcc1 = cv2.VideoWriter_fourcc(*'XVID')
        self.output_video_pth = ''#'/home/chfox/Documents/ARWAC/domReport/ref_images/Zedlivevideo.avi'
        # self.fourcc2 = cv2.VideoWriter_fourcc('M','J','P','G')
        # self.out = cv2.VideoWriter(self.output_video_pth,self.fourcc1, 10, (self.frame_width,self.frame_height))
        # self.out2 = cv2.VideoWriter('/home/dom/Documents/domReport/ref_images/OriAvi.mp4',fourcc2, 10, (frame_width,frame_height))
        self.piplineSelet = 4 # The algorithm selection : simpfliled with easy field 1, complex 2 : RGB and Depth map 3 

        # class CCamsParams (object):
        # def __init__(self,zed,sl):
        """        
        self.cam = None
        self.sl = None
        
        
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
        self.image_size.width = int(self.image_size.width)
        self.image_size.height =int(self.image_size.height)

        # Declare your sl.Mat matrices
        self.image_zed = self.sl.Mat(self.new_width, self.new_height, self.sl.MAT_TYPE.U8_C4)#sl.MAT_TYPE.MAT_TYPE_8U_C4)  # this for display view
        self.depth_image_zed = self.sl.Mat(self.new_width, self.new_height, self.sl.MAT_TYPE.U8_C4)#MAT_TYPE.MAT_TYPE_8U_C4) # this for the display view
        # image_zed = sl.Mat(image_size.width, image_size.height, sl.MAT_TYPE.U8_C4)
        # depth_image_zed = sl.Mat(image_size.width, image_size.height, sl.MAT_TYPE.U8_C4)
        self.point_cloud = self.sl.Mat()"""

        self.key = ' '
        self.depth_zed = None #self.sl.Mat(self.new_width, self.new_height, self.sl.MAT_TYPE.F32_C3)#.MAT_TYPE_32F_C3)
        # self.camera_pose = self.sl.Pose()
        self.viewer = ''#tv.PyTrackingViewer()
        # self.py_translation = self.sl.Translation()

        self.image_ocv = None
        self.frame_num = -1        
        
        # create runtime files
        self.Errfilepth = '/home/dom/Documents/ARWAC/robcontrl/04102021_video/'
        self.timestr = time.strftime("%Y%m%d-%H%M%S")
        # self.filesCreated2Save()

    def filesCreated2Save(self):
        # timestr = time.strftime("%Y%m%d-%H%M%S")
        # self.SenErrfilepth = '/home/dom/Documents/ARWAC/robdata/'
        # self.RevErrfilepth = '/home/dom/Documents/ARWAC/robdata/'  
        self.SenErrfilepthNm=os.path.join(self.Errfilepth, 'HoseOfSet_'+self.timestr+'.csv')
        # self.RevErrfilepthNm=os.path.join(self.Errfilepth, 'Recv_'+timestr+'.csv')
        
        return self.SenErrfilepthNm
        
        
    def perspective_transform(self,img_size,offset):
        
        """
        Execute perspective transform
        """
        # img_size = (img.shape[1], img.shape[0])
        # mask0 = np.zeros((img.shape[0], img.shape[1]), dtype="uint8")
        width = img_size[0]#int(img.shape[1] )
        height = img_size[1]# int(img.shape[0])
        # dim = (width, height)
        ############################################################# 
            
        pts = np.array([[0, height], [offset, 0], [width-offset, 0], [width, height]], dtype=np.int32)

        dst = np.float32(
            [[0, height],
            [0, 0],
            [width, 0],
            [width, height]])

        src = np.float32(
            [[0, height],
            [offset, 0],
            [width-offset, 0],
            [width, height]])
        m = cv2.getPerspectiveTransform(src, dst)
        m_inv = cv2.getPerspectiveTransform(dst, src)
        return m, m_inv


    def PltShowing(self,img,figname,titlname, pauseflg=True, dipFlag = True):    
        # paseflg=False
        if dipFlag == True:
            plt.figure(figname,figsize=(36,24))
            plt.title(titlname)
            plt.imshow(img,cmap='gray')
            # vmin=0, vmax=1)    for colorscale : the value ranage of colorscale of each map
            # plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB),cmap='gray', vmin=0, vmax=1)   
            plt.show(block=pauseflg)  
            plt.pause(0.25)
            plt.close()
        else:
            return   
    def cvSaveImages(self,img,figname='xx.jpg',path='./', svflg=False, subpth = '',extname='.jpg', covFlg = True):
        if svflg==True:            
            nampepath = os.path.join(self.svResPath, subpth, figname) 
            if covFlg == True:
                cv2.imwrite(nampepath,cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            else:
                cv2.imwrite(nampepath,img)
        else:
            return

    def convert(self,img, target_type_min, target_type_max, target_type):
        imin = img.min()
        imax = img.max()

        a = (target_type_max - target_type_min) / (imax - imin)
        b = target_type_max - a * imax
        new_img = (a * img + b).astype(target_type)
        return new_img

        """def __del__(self):
        if self.camera is not None:
            # self.cam.release()
            self.camera.close()
            self.camera = None
        if self.sl is not None:
            # self.sl.release()
            self.sl = None"""
        