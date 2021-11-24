#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pickle import TRUE
import sys
import numpy as np
import pyzed.sl as sl
import cv2
import matplotlib
matplotlib.use('TkAgg')
from tools.zedcvstream import zedcv # import zed processing methods
from tools.CSysParams import CLanesParams,CAlgrthmParams,CCamsParams
import matplotlib.pyplot as plt

import threading, math
# from tools.LaneDeteTrack import CLneDetsTracks
from tools.FullRGBLaneDeteTrack import CFullRGBLneDetsTracks
# from tools.FullRGBLaneDeteTrack_mix import CFullRGBLneDetsTracks
from argparse import ArgumentParser as ArgParse
import os
from tools.CSysParams import CLanesParams,CAlgrthmParams,CCamsParams#,CYoloV3Params
from tools.CInitialVideoSettings import CVideoSrcSetting,CCameraParamsSetting,CRobtoParamSetting
# from tools.CInitialVideoSettingsTX2 import CVideoSrcSetting,CCameraParamsSetting,CRobtoParamSetting

# from gt_metrics.Detector_GT import gt_detect
from tools.HSVRGBSegt4HT import PltShowing,cvSaveImages,HSVFullEdgeProc, RGBColorSegmtFull #, 
from tools.svo_export import ParsingZED
from tools.socketcom import CJetScketUDPClient #, CCtrlSocke
import enum
import ntpath
# help_string = "[s] Save side by side image [d] Save Depth, [n] Change Depth format, [p] Save Point Cloud, [m] Change Point Cloud format, [q] Quit"
# prefix_point_cloud = "Cloud_"
# prefix_depth = "Depth_"
#path = "./"
# path = "/home/dom/prototype/ZED/"
# count_save = 0
# mode_point_cloud = 0
# mode_depth = 0
# point_cloud_format = sl.POINT_CLOUD_FORMAT.POINT_CLOUD_FORMAT_XYZ_ASCII
# depth_format = sl.DEPTH_FORMAT.DEPTH_FORMAT_PNG
m_CLanParams = object()
m_CAgthmParams = object()
m_CamParams  = object()
# m_CLneDTrcks = object()
m_CFRGBLneTrcks = object()
m_CJetClnt = object()

def arwacmain(ArgAll):

    global m_CLanParams
    global m_CAgthmParams
    global m_CamParams 
    global m_CJetClnt
    m_CJetClnt = CJetScketUDPClient()
    # global m_CLneDTrcks
    global m_CFRGBLneTrcks
    m_CLanParams=CLanesParams() 
    m_VideoSrcSet = CVideoSrcSetting()
    #topdown
    ###################################################################################
    # src_vid = m_VideoSrcSet.data_data01022021svo_all() # George's 2nd farm , puddles, weather variant in middle
    # src_vid = m_VideoSrcSet.data_data01022021svo_com()
    # src_vid = m_VideoSrcSet.data07032021GeN1_com()
    src_vid = m_VideoSrcSet.data_26112020_4lanes_all()

    m_CLanParams.campitch = src_vid['pitch']
    m_CLanParams.video_file_l =   src_vid['path']#="/media/dom/Elements/data/TX2_Risehm/28072020/1st/goVL720q.avi"
    print("Reading AVI file: {0}".format(m_CLanParams.video_file_l))
    
    m_CLanParams.LaneOriPoint = src_vid['LaneOriPointL']#=50
    m_CLanParams.LaneEndPoint = src_vid['LaneEndPointR']#=50
    m_CLanParams.lanebandwith= src_vid['lanebandwidth']#=80

    m_CLanParams.Numoflanes = src_vid['NumofLanes']#=6  
    m_CLanParams.laneoffset= src_vid['laneoffset']#=155
    m_CLanParams.lanegap=int((m_CLanParams.LaneEndPoint-m_CLanParams.LaneOriPoint)/(m_CLanParams.Numoflanes-1)) 
    print("lane gap : {0}".format(m_CLanParams.lanegap))     

    m_CLanParams.laneStartPo = int(m_CLanParams.lanegap*0.5)
    if m_CLanParams.laneStartPo > int(m_CLanParams.lanebandwith*0.618):
        m_CLanParams.laneStartPo = int(m_CLanParams.lanebandwith*0.618) 
    # due  to the wide lane in this time
    # laneStartPo = int(lanegap*0.382)
    print("laneStartPo : {0}, lanegap : {1}, lanebandwith : {2}".format(m_CLanParams.laneStartPo, m_CLanParams.lanegap,m_CLanParams.lanebandwith)) 

    m_CLanParams.row0 =  src_vid['row0']#=350 image cropping top point
    m_CLanParams.col0 = int(m_CLanParams.LaneOriPoint-m_CLanParams.laneStartPo) # src_vid['col0']#=175  image cropping left point

    m_CLanParams.w0= int (m_CLanParams.LaneEndPoint-m_CLanParams.LaneOriPoint+m_CLanParams.laneStartPo*2)# int(src_vid['w0'])#=1025 total width of cropping image
    m_CLanParams.h0 = src_vid['frame_height'] # t   
    m_CLanParams.low_green  =src_vid['low_green']
    m_CLanParams.high_green = src_vid['high_green'] 
    

    print ('size of window row = {0}, width = {1} '.format(m_CLanParams.h0,m_CLanParams.w0))
    m_CLanParams.frm2strart = src_vid['frm2start']#=980

    m_CLanParams.avi_width =  m_CLanParams.w0#src_vid['frame_width']#=1025
    m_CLanParams.avi_height = m_CLanParams.h0#src_vid['frame_height']#=370    

    # Split filename and extension.    
    head, tail = ntpath.split(m_CLanParams.video_file_l)
    (name_pref, name_ext) = os.path.splitext(tail)
    upd_thrsh=20 # critirial for using kalman updting
    klman_start=-1# control of updating 'id' with kalman filter >30: #
    svfrmnum = 0 # indes for being saved image index , and detection positin in .csv  where the indext started from 1 in row
    # store errors 
    # TotalError1 = [] # coresponding to second lane in this case
    # TotalError3 = [] # coresponding to third lane in this case
    ############################################################
    # m_CLanParams.FullRGBRow0= row0#= 0
    # m_CLanParams.FullRGBH0=h0#=375
    # m_CLanParams.FullRGBCol0=col0# = 200
    # m_CLanParams.FullRGBW0=w0#=300

    # m_CLanParams.FullRGBbandwith=lanebandwith    #30 #estimation or setting the initial lane wide
    # m_CLanParams.FullRGBNumRowsInit=Numoflanes#.FullDTHNumRowsInit  #4
    # m_CLanParams.FullRGBoffset= laneoffset #25#25 #15  #25 
    # m_CLanParams.FullRGBLaneOriPoint=LaneOriPoint   # 25 # where the row started

    # lstartpt = LaneOriPoint + int (lanebandwith*0.5)
    # lanegap=int((snpwidth-Numoflanes*lanebandwith-LaneOriPoint*2)/(Numoflanes-1))
    m_CLanParams.deltalane= m_CLanParams.lanegap    
    m_CLanParams.PixelSize=m_CLanParams.lanegap  * m_CLanParams.mtricsize 
    ############################################################
    zed = sl.Camera()
    if zed.is_opened()==True:
        zed.close()
    # Set configuration parameters
    # init.camera_resolution = sl.RESOLUTION.RESOLUTION_HD1080
    # init.depth_mode = sl.DEPTH_MODE.DEPTH_MODE_ULTRA#.DEPTH_MODE_PERFORMANCE
    # init.coordinate_units = sl.UNIT.UNIT_METER
    # if len(sys.argv) >= 2 :
    #     init.svo_input_filename = sys.argv[1]
    input_type = sl.InputType()
    if len(sys.argv) >= 2 :
        input_type.set_from_svo_file(sys.argv[1])
        
    # init = sl.InitParameters(input_t=input_type)
    init = sl.InitParameters()
    init.camera_resolution = sl.RESOLUTION.HD720#.HD1080
    init.depth_mode = sl.DEPTH_MODE.PERFORMANCE
    init.coordinate_units = sl.UNIT.METER#MILLIMETER
    init.depth_minimum_distance=0.5#minimum depth perception distance to half meter
    init.depth_maximum_distance=20#maxium depth percetion distan to 20 meter
    init.camera_fps=30#set fps at 30

    init.set_from_svo_file(str(m_CLanParams.video_file_l))
    init.svo_real_time_mode = False  # Don't convert in realtime

    # Open the camera
    err = zed.open(init)
    if err != sl.ERROR_CODE.SUCCESS :
        print(repr(err))
        zed.close()
        # exit(1)

   # Get image size
    # image_size = zed.get_camera_information().camera_resolution
    # width = image_size.width
    # height = image_size.height
    # width_sbs = width * 2
    m_CamParams = CCamsParams(zed, sl)
    # Prepare side by side image container equivalent to CV_8UC4
    # svo_image_sbs_rgba = np.zeros((height, width_sbs, 4), dtype=np.uint8)

    # Prepare single image containers
    # m_CamParams.image_size = image_size
    m_CamParams.left_image = sl.Mat()
    m_CamParams.right_image = sl.Mat()
    m_CamParams.depth_image = sl.Mat()
    m_CamParams.rt_param = sl.RuntimeParameters()
    m_CamParams.rt_param.sensing_mode = sl.SENSING_MODE.FILL

    # Start SVO conversion to AVI/SEQUENCE
    sys.stdout.write("Converting AVI... Use Ctrl-C to interrupt conversion.\n")

    # key = ' '

    """
    prefix_point_cloud = "Cloud_"
    prefix_depth = "Depth_"
    #path = "./"
    path = "/home/chfox/prototype/ZED/"
    count_save = 0
    mode_point_cloud = 0
    mode_depth = 0
    point_cloud_format = sl.POINT_CLOUD_FORMAT.POINT_CLOUD_FORMAT_XYZ_ASCII
    depth_format = sl.DEPTH_FORMAT.DEPTH_FORMAT_PNG
    https://www.stereolabs.com/docs/opencv/python/

    """  

    
    # m_CamParams.totalNum_frame = zed.get_svo_number_of_frames()
    # print('total number of frames in .svo files := ', m_CamParams.totalNum_frame) 
    # m_CLanParams.FullRGBNumRowsInit = Numoflanes
    video_ocv =   src_vid['path']#="/media/dom/Elements/data/TX2_Risehm/28072020/1st/goVL720q.avi"
    video_dpt =   src_vid['path_dpt']#="/media/dom/Elements/data/TX2_Risehm/28072020/1st/goVL720q.avi"
   
    outputpth ='/home/dom/Documents/ARWAC/domReport/ref_images/Zedlivevideo.avi'
    m_CAgthmParams = CAlgrthmParams(zed,sl,video_ocv,video_dpt,outputpth)
    # m_CLanParams.FullRRGNumRowsInit = ArgAll.numL # initial lane settting of numbers for tracking
    m_CAgthmParams.piplineSelet =4# ArgAll.pipL # Select different algorithm for detection lanes 1: for early tidy lane, 2: RGB highly adjoining lane, 3: Depth map on 2
    # print ('Num of Lanes Setting = ', ArgAll.numL)
    #Algorithms encapsulatd here:
    # m_CLneDTrcks = CLneDetsTracks(m_CAgthmParams,m_CLanParams) 
    m_CFRGBLneTrcks = CFullRGBLneDetsTracks(m_CAgthmParams,m_CLanParams)
    m_zedcvHandle = zedcv(zed,sl,m_CLanParams.video_file_l) # instance of ZED stream
    # start_zed(zed, runtime, camera_pose, viewer, py_translation,depth_zed,image_zed,depth_image_zed,new_width,new_height,point_cloud,key)
    m_CLanParams.mt,m_CLanParams.mt_inv = m_CFRGBLneTrcks.perspective_transform((m_CLanParams.w0,m_CLanParams.h0),m_CLanParams.laneoffset)

    start_zed()
    
    

# def start_zed(m_CamParams):
#     zed_callback = threading.Thread(target=run, args=(zed, runtime, camera_pose, viewer, py_translation,depth_zed,image_zed,depth_image_zed,new_width,new_height,point_cloud,key))
#     zed_callback.start()

def start_zed():
    global m_CamParams
     # zed_callback = threading.Thread(target=run, args=(m_CamParams.cam, m_CamParams.sl))
    stop_threads = False
    zed_callback = threading.Thread(target=run,args =(lambda : stop_threads,)) # must be iterable so ','
    zed_callback.start()

# def run(cam, runtime, camera_pose, viewer, py_translation):
def run(killed):       
    global m_CamParams
    # global m_CLneDTrcks
    global m_CAgthmParams
    global m_CFRGBLneTrcks
    global m_CLanParams
     #open camera for streaming , temporarily for the test on video files collected
    # m_CLneDTrcks.CameraStreamfromfile()
    # m_CFRGBLneTrcks.CameraStreamfromfile() # for .avi files
    # m_CLneDTrcks.MainDeteTrack()#("/home/chfox/ARWAC/Essa-0/20190206_143625.mp4")
    m_CAgthmParams.issvo = False
    m_CAgthmParams.isavi = True
    m_CamParams.image_ocv = None
    m_CamParams.frame_num = -1
    msg = {}
    # m_CamParams.cam = 
    
    m_CFRGBLneTrcks.CameraStreamfromfile()
    while m_CamParams.key != 113 : #F2
        if killed() == True:
            print('Terminated...!')
            break

        # using .svo files
        if m_CAgthmParams.isavi == True:

            #############################
            grabbed, orig_frame = m_CAgthmParams.camera.read()
            if (grabbed is False):
                print ("failed to grab frame.")
                camera = cv2.VideoCapture(0)
                continue

            m_CamParams.frame_num=m_CamParams.frame_num+1
            if m_CamParams.frame_num<=m_CLanParams.frm2strart:
                continue

            # Check if we have reached the end of the video
          
            if m_CamParams.frame_num%1!=0:
                continue
            # Retrieve SVO images
            # zed.retrieve_image(left_image, sl.VIEW.LEFT)
            m_CAgthmParams.filter_start = m_CAgthmParams.filter_start +1
            # m_CamParams.cam.retrieve_image(m_CamParams.left_image, sl.VIEW.LEFT, sl.MEM.CPU,m_CamParams.image_size)
                       
            # orig_frame = m_CamParams.left_image.get_data()
            # Convert SVO image from RGBA to RGB
            # orig_frame = cv2.cvtColor(orig_frame, cv2.COLOR_RGBA2RGB)
            # orig_frame = cv2.cvtColor(orig_frame, cv2.COLOR_RGB2BGR)
            # plt.figure('AVI File')
            # plt.title('Visible and Depth')        
            # plt.imshow(orig_frame,cmap='gray')
            # plt.show(block=False)
            # plt.pause(0.05)
            # plt.close()           
            
            RGBPiplines(orig_frame) 
            msgi = m_CLanParams.Rbt_Ofset 
            # msgo = sum(m_CLanParams.Ctrl_Ofset.values())/(m_CLanParams.Numoflanes*m_CLanParams.deltalane)
            # print ('Robot control offset {inter} and {externa}'.format(inter = msgi, externa = msgo))         
            # scaled by lane width
            msg_scl = msgi/m_CLanParams.deltalane
            msg_scl= msg_scl*10
            if abs(msg_scl)>1:
                msg_scl=0.0
            print ('Robot control offset {inter} and {scaled by gap}'.format(inter = msgi, externa = msg_scl))

            angle_to_eveg_radian = math.atan(msgi / 120.0)  # half window
            angle_to_eveg_radian = math.atan(msgi / 240.0)  # whole window

            msg = msgi*m_CLanParams.PixelSize  # convet pixle to physical size (cm)
            angle_to_eveg_radian = math.atan(msg / 600.0)  # navigation point to center of rear track/bodybase 


            angle_to_ever_deg = int(angle_to_eveg_radian * 180.0 / math.pi)  
            steering_angle = angle_to_ever_deg + 90 
            print ('Robot steering radian {inter} and degree {externa}'.format(inter = angle_to_eveg_radian, externa = angle_to_ever_deg))

            m_CJetClnt.Data2Send(angle_to_eveg_radian)

            # m_CJetClnt.Data2Send(msgi)
            print('\n')
            print("Sending out : {0}".format(angle_to_eveg_radian))
            continue

        # using live video

    plt.close('all')
    cv2.destroyAllWindows()
    m_CamParams.cam.close()
    print("\nFINISH")


def RGBPiplines(img):   
    # plt.figure('ZED Camera Live')
    # plt.title('Visible and Depth')        
    # plt.imshow(img,cmap='gray')
    # plt.show(block=False)
    # plt.pause(0.25)
    # plt.close()
    m_CFRGBLneTrcks.MainDeteTrack(img) 

def get_parser():
    parser =ArgParse()
    parser.add_argument('--numL', type=int, help="Initial Setting of total number of Lane to track !",default=4)
    parser.add_argument('--pipL', type=int, help="Detecting Pipe Line Choosing: simplified 1, complex- RGB (2) and Depth (3) !",default=2)

    return parser


if __name__ == "__main__":

    args =get_parser().parse_args()
    arwacmain(args)
