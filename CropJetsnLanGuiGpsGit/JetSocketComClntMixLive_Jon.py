#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pickle import TRUE
import sys
# import numpy as np
from numpy.core.numeric import True_
import pyzed.sl as sl
import cv2
from tools.zedcvstream import zedcv # import zed processing methods
from tools.CSysParams import CLanesParams,CAlgrthmParams,CCamsParams
import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import threading,signal
# from tools.LaneDeteTrack import CLneDetsTracks
# from tools.FullRGBLaneDeteTrack import CFullRGBLneDetsTracks
from tools.FullRGBLaneDeteTrack_mix import CFullRGBLneDetsTracks
from argparse import ArgumentParser as ArgParse
import os
from tools.CSysParamsAll import CLanesParams#,CAlgrthmParams,CCamsParams#,CYoloV3Params
# from CInitialVideoSettings_lbl import CVideoSrcSetting,CCameraParamsSetting,CRobtoParamSetting
from tools.CInitialVideoSettingsLive import CVideoSrcSetting,CCameraParamsSetting,CRobtoParamSetting

# from gt_metrics.Detector_GT import gt_detect
from tools.HSVRGBSegt4HT import PltShowing,cvSaveImages,HSVFullEdgeProc, RGBColorSegmtFull #, 
from tools.svo_export import ParsingZED
# from tools.socketcom import CJetScketUDPClient #, CCtrlSocke
import math, enum
import ntpath,time
from ARWACv2dom.mavlink_ctrl import CMavlinkCtrl
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
m_MavlinkCtrl = object()
m_CLanParams = object()
# m_CAgthmParams = object()
# m_CamParams  = object()
# m_CLneDTrcks = object()
m_CFRGBLneTrcks = object()
# m_CJetClnt = object()

import readchar


# https://code-maven.com/catch-control-c-in-python
def signal_handler(signal, frame):
    msg = "Ctrl-c was pressed. Do you really want to exit? y/n "
    print(msg, end="", flush=True)
    res = readchar.readchar()
    if res == 'y':
        print("")
        exit(1)
    else:
        print("", end="\r", flush=True)
        print(" " * len(msg), end="", flush=True) # clear the printed line
        print("    ", end="\r", flush=True)
        
    # sys.exit(1)

signal.signal(signal.SIGINT, signal_handler)

def arwacmain(ArgAll):

    global m_CLanParams
    # global m_CAgthmParams
    # global m_CamParams 
    global m_CFRGBLneTrcks
    # global m_CJetClnt

    global m_MavlinkCtrl
    m_MavlinkCtrl = CMavlinkCtrl()

    # m_CJetClnt = CJetScketUDPClient()
    # global m_CLneDTrcks
    
    
    m_CLanParams=CLanesParams() 

    m_CLanParams.camera = sl.Camera()
    if m_CLanParams.camera.is_opened()==True:
        m_CLanParams.camera.close()

    m_VideoSrcSet = CVideoSrcSetting()
    #topdown
    ###################################################################################
    # src_vid = m_VideoSrcSet.data_05112020_4lanes_all()
    # src_vid = m_VideoSrcSet.data_12112020_4lanes_all()
    # src_vid = m_VideoSrcSet.data_19112020_4lanes_all()
    # src_vid = m_VideoSrcSet.data_26112020_4lanes_all()
    # src_vid = m_VideoSrcSet.data_26112020_svo_all()
    # src_vid = m_VideoSrcSet.data_10122020_svo_all()
    # src_vid = m_VideoSrcSet.data_data22012021_svo_all() # George's 2nd farm , puddles, weather variant in middle
    # src_vid = m_VideoSrcSet.data_data01022021svo_all() # George's 2nd farm , puddles, weather variant in middle

    src_vid = m_VideoSrcSet.data_live_setting() # George's 2nd farm , puddles, weather variant in middle

    m_CLanParams.campitch = src_vid['pitch']
    m_CLanParams.video_file_l =   src_vid['path']#="/media/dom/Elements/data/TX2_Risehm/28072020/1st/goVL720q.avi"
    # print("Reading SVO file: {0}".format(m_CLanParams.video_file_l))
    
    m_CLanParams.LaneOriPoint = src_vid['LaneOriPointL']#=50
    m_CLanParams.LaneEndPoint = src_vid['LaneEndPointR']#=50
    m_CLanParams.lanebandwith= src_vid['lanebandwidth']#=80

    m_CLanParams.Numoflanes = src_vid['NumofLanes']#=6  
    m_CLanParams.laneoffset= src_vid['laneoffset']#=155
    m_CLanParams.lanegap=int((m_CLanParams.LaneEndPoint-m_CLanParams.LaneOriPoint)/(m_CLanParams.Numoflanes-1)) 
    # print("lane gap : {0}".format(m_CLanParams.lanegap))     

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
    # head, tail = ntpath.split(m_CLanParams.video_file_l)
    # (name_pref, name_ext) = os.path.splitext(tail)
    # upd_thrsh=20 # critirial for using kalman updting
    # klman_start=-1# control of updating 'id' with kalman filter >30: #
    # svfrmnum = 0 # indes for being saved image index , and detection positin in .csv  where the indext started from 1 in row

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
    m_CLanParams.PixelSize=m_CLanParams.lanegap  / m_CLanParams.mtricsize  
    ############################################################
    # zed = sl.Camera()
    # if zed.is_opened()==True:
    #     zed.close()
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
    init.depth_minimum_distance=0.25#minimum depth perception distance to half meter
    init.depth_maximum_distance=20#maxium depth percetion distan to 20 meter
    init.camera_fps=15#set fps at 30

    # init.set_from_svo_file(str(m_CLanParams.video_file_l))
    # init.svo_real_time_mode = False  # Don't convert in realtime

    # Open the camera
    err = m_CLanParams.camera.open(init)
    if err != sl.ERROR_CODE.SUCCESS :
        print(repr(err))
        m_CLanParams.camera.close()
        # exit(1)

   # Get image size
    # image_size = zed.get_camera_information().camera_resolution
    # width = image_size.width
    # height = image_size.height
    # width_sbs = width * 2
    # m_CamParams = CCamsParams(zed, sl)
    # Prepare side by side image container equivalent to CV_8UC4
    # svo_image_sbs_rgba = np.zeros((height, width_sbs, 4), dtype=np.uint8)

    # Prepare single image containers
    # m_CamParams.image_size = image_size
    m_CLanParams.left_image = sl.Mat()
    # m_CamParams.right_image = sl.Mat()
    # m_CamParams.depth_image = sl.Mat()
    m_CLanParams.rt_param = sl.RuntimeParameters()
    m_CLanParams.rt_param.sensing_mode = sl.SENSING_MODE.FILL
    m_CLanParams.image_size = m_CLanParams.camera.get_camera_information().camera_resolution#get_resolution()

    # Start SVO conversion to AVI/SEQUENCE
    # sys.stdout.write("Converting SVO... Use Ctrl-C to interrupt conversion.\n")



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
   
    outputpth =''#'/home/dom/Documents/ARWAC/domReport/ref_images/Zedlivevideo.avi'
    # m_CAgthmParams = CAlgrthmParams(zed,sl,video_ocv,video_dpt,outputpth)
    # m_CLanParams.FullRRGNumRowsInit = ArgAll.numL # initial lane settting of numbers for tracking
    # m_CAgthmParams.piplineSelet = ArgAll.pipL # Select different algorithm for detection lanes 1: for early tidy lane, 2: RGB highly adjoining lane, 3: Depth map on 2
    # print ('Num of Lanes Setting = ', ArgAll.numL)
    #Algorithms encapsulatd here:
    # m_CLneDTrcks = CLneDetsTracks(m_CAgthmParams,m_CLanParams) 
    m_CFRGBLneTrcks = CFullRGBLneDetsTracks(m_CLanParams)
    # m_zedcvHandle = zedcv(zed,sl,m_CLanParams.video_file_l) # instance of ZED stream
    # start_zed(zed, runtime, camera_pose, viewer, py_translation,depth_zed,image_zed,depth_image_zed,new_width,new_height,point_cloud,key)
    m_CLanParams.mt,m_CLanParams.mt_inv = m_CFRGBLneTrcks.perspective_transform((m_CLanParams.w0,m_CLanParams.h0),m_CLanParams.laneoffset)
    start_zed()
    
    

# def start_zed(m_CamParams):
#     zed_callback = threading.Thread(target=run, args=(zed, runtime, camera_pose, viewer, py_translation,depth_zed,image_zed,depth_image_zed,new_width,new_height,point_cloud,key))
#     zed_callback.start()

def start_zed():
    global m_CLanParams
     # zed_callback = threading.Thread(target=run, args=(m_CamParams.cam, m_CamParams.sl))
    stop_threads = False
    zed_callback = threading.Thread(target=run,args =(lambda : stop_threads,)) # must be iterable so ','
    zed_callback.start()

# def run(cam, runtime, camera_pose, viewer, py_translation):
def run(killed):       
    # global m_CLneDTrcks
    # global m_CAgthmParams
    global m_CFRGBLneTrcks
    global m_CLanParams
    global m_MavlinkCtrl
     #open camera for streaming , temporarily for the test on video files collected
    # m_CLneDTrcks.CameraStreamfromfile()
    # m_CFRGBLneTrcks.CameraStreamfromfile() # for .avi files
    # m_CLneDTrcks.MainDeteTrack()#("/home/chfox/ARWAC/Essa-0/20190206_143625.mp4")
    m_CLanParams.issvo = False
    m_CLanParams.isavi = False
    m_CLanParams.image_ocv = None
    m_CLanParams.frame_num = -1


    sim_out = 0.0
    last_out = sim_out
    sleepT = 0.1
    
    rb_sleepT = 0.0
    avg_value=0.0
    msg_scl=0.0
    msg_vec = list()
    
    scale = -10.0 # control the normalization
    ctrFlag = 1 # selection of methods
    Lbase=200.0 #cm
    Ibase = 240 # pixes : height detection windows size
    Wbase = Ibase**m_CLanParams.PixelSize # approximately , pixel is squared - detection windows height

    while m_CLanParams.key != 113 : #F2
        start_time = time.time()
        if killed() == True:
            print('Terminated...!')
            break
        
        # using .svo files
        if m_CLanParams.issvo == TRUE or m_CLanParams.isavi ==True:

            #############################
            if m_CLanParams.cam.grab(m_CLanParams.rt_param) != sl.ERROR_CODE.SUCCESS:
                print ('filed to extract frame in .svo file')
                # print(repr(status))
                break

            # svo_position = m_CamParams.cam.get_svo_position()
            m_CLanParams.frame_num=m_CLanParams.frame_num+1
            if m_CLanParams.frame_num<=m_CLanParams.frm2strart:
                continue

            # # Check if we have reached the end of the video
            # if svo_position >= (m_CamParams.totalNum_frame - 1):  # End of SVO
            #     sys.stdout.write("\nSVO end has been reached. Exiting now.\n")
            #     break 
            
            # if m_CamParams.frame_num%1!=0:
                # continue
            # Retrieve SVO images
            # zed.retrieve_image(left_image, sl.VIEW.LEFT)
            m_CLanParams.cam.retrieve_image(m_CLanParams.left_image, sl.VIEW.LEFT, sl.MEM.CPU,m_CLanParams.image_size)
            """
            if app_type == AppType.LEFT_AND_RIGHT:
                zed.retrieve_image(m_CamParams.right_image, sl.VIEW.RIGHT)
            elif app_type == AppType.LEFT_AND_DEPTH:
                zed.retrieve_image(m_CamParams.right_image, sl.VIEW.DEPTH)
            elif app_type == AppType.LEFT_AND_DEPTH_16:
                zed.retrieve_measure(m_CamParams.depth_image, sl.MEASURE.DEPTH)
            """
            # Generate file names
            # filename = "left%s.png" % str(svo_position).zfill(6)
            # filename1 = svRes / ("left%s.png" % str(svo_position).zfill(6))
            # filename2 = svRes / (("right%s.png" if app_type == AppType.LEFT_AND_RIGHT
                                        # else "depth%s.png") % str(svo_position).zfill(6))

            # Save Left images
            # cv2.imwrite(str(filename1), left_image.get_data())

            # ret, orig_frame = video.read()
            
            orig_frame = m_CLanParams.left_image.get_data()
            # Convert SVO image from RGBA to RGB
            orig_frame = cv2.cvtColor(orig_frame, cv2.COLOR_RGBA2RGB)
            orig_frame = cv2.cvtColor(orig_frame, cv2.COLOR_RGB2BGR)
            plt.figure('ZED SVO File')
            plt.title('Visible and Depth')        
            plt.imshow(orig_frame,cmap='gray')
            plt.show(block=True)
            plt.pause(0.05)
            plt.close()           
            
            # RGBPiplines(orig_frame) 
            m_CFRGBLneTrcks.MainDeteTrack(orig_frame) 
            # testing sending message out
            # msg['L'] = 555.0
            # msg['R'] = 999.0
            # msg['N'] = 0.0
            msgi = m_CLanParams.Rbt_Ofset 
            
            # scaled by lane width
            msg_scl = msgi/m_CLanParams.deltalane
            print ('Robot control offset {inter} and {scaled by gap}'.format(inter = msgi, externa = msg_scl))
            # m_CJetClnt.Data2Send(msg_scl)
            plt.pause(0.05)
            continue
            angle_to_eveg_radian = math.atan(msgi / 120.0)  # half window
            angle_to_eveg_radian = math.atan(msgi / 240.0)  # whole window
            m_CJetClnt.Data2Send(angle_to_eveg_radian)
            plt.pause(0.05)
            continue
            msg = msgi*m_CLanParams.PixelSize  # convet pixle to physical size (cm)
            angle_to_eveg_radian = math.atan(msg / 600.0)  # navigation point to center of rear track/bodybase 
            m_CJetClnt.Data2Send(angle_to_eveg_radian)
            plt.pause(0.05)
            continue

            angle_to_ever_deg = int(angle_to_eveg_radian * 180.0 / math.pi)  
            steering_angle = angle_to_ever_deg + 90 
            print ('Robot steering radian {inter} and degree {externa}'.format(inter = angle_to_eveg_radian, externa = angle_to_ever_deg))

            m_CJetClnt.Data2Send(angle_to_eveg_radian)
            plt.pause(0.05)
            continue

        # using live video

        err = m_CLanParams.camera.grab(m_CLanParams.rt_param)
        if err == sl.ERROR_CODE.SUCCESS :
                               
            # m_CamParams.cam.retrieve_measure(m_CamParams.depth_zed, m_CamParams.sl.MEASURE.DEPTH) # application purpose
            # Load depth data into a numpy array
            # m_CamParams.depth_ocv = m_CamParams.depth_zed.get_data()
            # Print the depth value at the center of the image
            # print('center of image = ', m_CamParams.depth_ocv[int(len(m_CamParams.depth_ocv)/2)][int(len(m_CamParams.depth_ocv[0])/2)])

            # m_CamParams.cam.retrieve_image(m_CamParams.image_zed, m_CamParams.sl.VIEW.LEFT) # Left image 
            # Use get_data() to get the numpy array === full
            # m_CamParams.image_ocv = m_CamParams.image_zed.get_data()
            # report = np.hstack((image_ocv,depth_zed)) #stacking images side-by-side
            # Display the left image from the numpy array
            # cv2.imshow("Image", image_ocv)
            # plt.figure('ZED Camera Live')
            # plt.title('depth_zed')        
            # plt.imshow(m_CamParams.depth_ocv,cmap='gray')
            # plt.show(block=False)
            # plt.pause(0.05)
            # plt.close()
            # Retrieve the left image, depth image in the half-resolution ----------- only for the display purpose
            # m_CamParams.cam.retrieve_image(m_CamParams.image_zed, m_CamParams.sl.VIEW.VIEW_LEFT, m_CamParams.sl.MEM.MEM_CPU, int(m_CamParams.new_width), int(m_CamParams.new_height))
            # m_CamParams.cam.retrieve_image(m_CamParams.depth_image_zed, m_CamParams.sl.VIEW.VIEW_DEPTH, m_CamParams.sl.MEM.MEM_CPU, int(m_CamParams.new_width), int(m_CamParams.new_height))
            m_CLanParams.camera.retrieve_image(m_CLanParams.left_image, sl.VIEW.LEFT, sl.MEM.CPU, m_CLanParams.image_size)#(m_CamParams.new_width, m_CamParams.new_height))
            # m_CamParams.cam.retrieve_image(m_CamParams.depth_image_zed, m_CamParams.sl.VIEW.DEPTH, m_CamParams.sl.MEM.CPU, m_CamParams.image_size)#(m_CamParams.new_width, m_CamParams.new_height))
                       
            # To recover data from sl.Mat to use it with opencv, use the get_data() method
            # It returns a numpy array that can be used as a matrix with opencv
            m_CLanParams.image_ocv = m_CLanParams.left_image.get_data()
            # m_CamParams.depth_image_ocv = m_CamParams.depth_image_zed.get_data()

            # report = np.hstack((image_ocv,depth_ocv)) #stacking images side-by-side
            # report = np.hstack((m_CamParams.image_ocv,m_CamParams.depth_image_ocv)) #stacking images side-by-side
            # cv2.imwrite('/home/chfox/Documents/ARWAC/FarmLanedetection/ref_images/Fig18EdgsDepth.jpg',regionofintrest)
            # cv2.imwrite('/home/chfox/Documents/ARWAC/FarmLanedetection/ref_images/Fig18HoughLineDepth.jpg',masked)
            # cv2.imwrite('/home/chfox/Documents/ARWAC/FarmLanedetection/ref_images/Fig18EdgeandHhLinesDepth.jpg',report)
            # plt.figure('ZED Camera Live')
            # plt.title('Visible and Depth')        
            # plt.imshow(report,cmap='gray')
            # plt.show(block=False)
            # plt.pause(0.05)
            # plt.close()
            # if m_CAgthmParams.piplineSelet==1:
            #     SimpfiedRGBPiplines(m_CamParams.image_ocv)
            #     pass
            # elif m_CAgthmParams.piplineSelet==2:
                # RGBPiplines(m_CamParams.image_ocv)     
                # pass    
                # 
                # 
                # 
                #     
            
            orig_frame = cv2.cvtColor(m_CLanParams.image_ocv, cv2.COLOR_RGBA2RGB)
            orig_frame = cv2.cvtColor(orig_frame, cv2.COLOR_RGB2BGR)
            # plt.figure('ZED Real Time')
            # plt.title('Visible')        
            # plt.imshow(orig_frame,cmap='gray')
            # plt.show(block=True)
            # plt.pause(0.05)
            # plt.close()           
            m_CLanParams.frame_num=m_CLanParams.frame_num+1
            if m_CLanParams.frame_num<3:#<=m_CLanParams.frm2strart:
                continue 
            # RGBPiplines(orig_frame) 
            m_CFRGBLneTrcks.MainDeteTrack(orig_frame) 
            # testing sending message out
            # msg['L'] = 555.0
            # msg['R'] = 999.0
            # msg['N'] = 0.0
                    
            msgi = m_CLanParams.Rbt_Ofset  # lane offset convert to control signal
           
            if ctrFlag == 1:
                # scaled by lane width
                msg_scl = scale*msgi/m_CLanParams.deltalane
                print ('Robot control offset {inter} and scaled as {externa}'.format(inter = msgi, externa = msg_scl))                
                                
            elif ctrFlag == 2:
                msg_scl = 0.5*scale*(msgi / Ibase)  # angle_to_eveg_radian:  half window
                print ('Robot control offset {inter} and scaled by half window {externa}'.format(inter = msgi, externa = msg_scl))

            elif ctrFlag == 3:
                msg_scl = scale*(msgi / Ibase)  # angle_to_eveg_radian  whole window
                print ('Robot control offset {inter} and scaled by detection window {externa}'.format(inter = msgi, externa = msg_scl))

            elif ctrFlag == 4:
                msg = msgi*m_CLanParams.PixelSize  # convet pixle to physical size (cm)
                angle_to_eveg_radian = msg / Lbase  # not using window height, turing point is imeditely , otherwise ( (Lbase +Wbase))
                msg_scl = scale*angle_to_eveg_radian  # angle_to_eveg_radian - navigation point to center of rear track/bodybase 
                print ('Robot steering radian {inter} and scaled by pixels and robot body {externa}'.format(inter = angle_to_eveg_radian, externa = msg_scl))

            elif ctrFlag == 5:
                msg = msgi*m_CLanParams.PixelSize  # convet pixle to physical size (cm)
                angle_to_eveg_radian = angle_to_eveg_radian = 0.5*msg / Lbase  # angle_to_eveg_radian  navigation point to center of rear track/bodybase 
                angle_to_ever_deg = angle_to_eveg_radian * 180.0 /math.pi  # angle_to_ever_deg - -degree
                
                steering_angle = angle_to_ever_deg + 90 # taking into the direction...............
                msg_scl = scale*angle_to_ever_deg / 180.0  # angle_to_ever_deg - -normalized by 180 degree                               
                print ('Robot steering radian {inter} and scaled by steering degree {externa}'.format(inter = angle_to_eveg_radian, externa = msg_scl))

            elif ctrFlag == 6:
                msg = msgi*m_CLanParams.PixelSize  # convet pixle to physical size (cm)
                # angle_to_eveg_radian = math.atan(msg / Lbase)  # angle_to_eveg_radian  navigation point to center of rear track/bodybase 
                angle_to_eveg_radian = msg / Lbase
                st_ang = math.atan(2*Lbase*math.sin(angle_to_eveg_radian)/msg) # lateral offset: msg = look-ahaed distance 

                angle_to_ever_deg = st_ang * 180.0 /math.pi  # angle_to_ever_deg - -degree
                
                steering_angle = angle_to_ever_deg + 90 # taking into the direction...............
                msg_scl = scale*st_ang / 180.0  # angle_to_ever_deg - -normalized by 180 degree                               
                print ('Robot steering radian {inter} and scaled by steering degree {externa}'.format(inter = angle_to_eveg_radian, externa = msg_scl))

            if abs(msg_scl)>=1:
                msg_scl=0.618*msg_scl/abs(msg_scl)

            # if msg_scl>0.0:
            #     msg_scl =0.95
            # elif msg_scl <0.0:
            #     msg_scl= -0.95
            msg_scl = round(msg_scl,3)
            """
            # using every 15 - 60 frames average as ouput
            msg_vec.append(msg_scl)
            if m_CamParams.frame_num %15==0:
                avg_value = sum(msg_vec) / len(msg_vec)
                sim_out=avg_value
                msg_vec=[]
            """
            # by the time category only
            # optiion 1:
            # if abs(last_out)>0.75:
            #     if rb_sleepT>=3.5:
            #         # sim_out=avg_value#msg_scl
            #         sim_out=msg_scl
            #         rb_sleepT=0.0

            # elif abs(last_out)>0.5:
            #     if rb_sleepT>=3.0:
            #         # sim_out=avg_value#msg_scl
            #         sim_out=msg_scl
            #         rb_sleepT=0.0

            # elif abs(last_out)>0.25:
            #     if rb_sleepT>=2.5:
            #         # sim_out=avg_value#msg_scl
            #         sim_out=msg_scl
            #         rb_sleepT=0.0

            # elif abs(last_out)>=0.0:
            #     if rb_sleepT>=2.0:
            #         # sim_out=avg_value#msg_scl
            #         sim_out=msg_scl
            #         rb_sleepT=0.0
            ############################

            # # option 2:
            # if abs(last_out)>0.75:
            #     if rb_sleepT>=1.0:
            #         # sim_out=avg_value #msg_scl
            #         sim_out=msg_scl
            #         rb_sleepT=0.0

            # elif abs(last_out)>0.5:
            #     if rb_sleepT>=1.50:
            #         # sim_out=avg_value#msg_scl
            #         sim_out=msg_scl
            #         rb_sleepT=0.0

            # elif abs(last_out)>0.25:
            #     if rb_sleepT>=2.0:
            #         # sim_out=avg_value#msg_scl
            #         sim_out=msg_scl
            #         rb_sleepT=0.0

            # elif abs(last_out)>=0.0:
            #     if rb_sleepT>=2.5:
            #         # sim_out=avg_value#msg_scl
            #         sim_out=msg_scl
            #         rb_sleepT=0.0
            # ############################

            # option 3:
            ############################
            # if abs(last_out)<0.25:
            #     if rb_sleepT>=2.0:
            #         # sim_out=avg_value#msg_scl
            #         sim_out=msg_scl
            #         rb_sleepT=0.0

            # elif abs(last_out)<0.5:
            #     if rb_sleepT>=2.5:
            #         # sim_out=avg_value#msg_scl
            #         sim_out=msg_scl
            #         rb_sleepT=0.0

            # elif abs(last_out)<0.75:
            #     if rb_sleepT>=3.0:
            #         # sim_out=avg_value#msg_scl
            #         sim_out=msg_scl
            #         rb_sleepT=0.0

            # elif abs(last_out)<=1.0:
            #     if rb_sleepT>=3.5:
            #         # sim_out=avg_value#msg_scl
            #         sim_out=msg_scl
            #         rb_sleepT=0.0
            #####################
            #####################
            """
            if rb_sleepT>=1.5:
                # #sim_out=avg_value#msg_scl
                sim_out=msg_scl#+np.random.rand()/5.0
                rb_sleepT=0.0
            """
            #####################
            #####################
            sim_out=msg_scl            
            # m_CJetClnt.Data2Send(sim_out)
            m_MavlinkCtrl.steering_receive_camera(sim_out)
            last_out = sim_out 
            # time.sleep(sleepT)
            rb_sleepT = rb_sleepT+sleepT
            secs = math.floor((time.time() - start_time))       
            # print("--- %s seconds ---" % (time.time() - start_time))         
            print ('Live time Running: {seTime} seconds for Steering:  {externa}'.format(seTime =secs, externa = sim_out))


            # elif m_CAgthmParams.piplineSelet==3:
                # DepthMapPiplines(m_CamParams.image_ocv)
                # pass                  

            # switch_pipline(m_CAgthmParams.piplineSelet,m_CamParams.image_ocv)        

            # Retrieve the RGBA point cloud in half resolution           
            # m_CamParams.cam.retrieve_measure(m_CamParams.point_cloud, m_CamParams.sl.MEASURE.MEASURE_XYZRGBA, m_CamParams.sl.MEM.MEM_CPU, int(m_CamParams.new_width), int(m_CamParams.new_height)) 

            """
            ################################# ------------- save as image
            filename = path + prefix_depth + str(count_save)
            print('filename = {0}'.format(filename))
            m_zedcvHandle.save_depth(zed, filename)           
            
            mode_depth += 1
            depth_format = sl.DEPTH_FORMAT(mode_depth % 3)
            print("Depth format: ", m_zedcvHandle.get_depth_format_name(depth_format))
            m_zedcvHandle.save_point_cloud(zed, path + prefix_point_cloud + str(count_save))
            mode_point_cloud += 1
            point_cloud_format = sl.POINT_CLOUD_FORMAT(mode_point_cloud % 4)
            print("Point Cloud format: ", m_zedcvHandle.get_point_cloud_format_name(point_cloud_format))

            m_zedcvHandle.save_sbs_image(zed, path + "ZED_image" + str(count_save) + ".png")

            depth_image_ocv = np.concatenate((image_ocv, depth_image_ocv), axis=1)
            m_zedcvHandle.save_depth_ocv(path + "ZED_image_Depth" + str(count_save) + ".png",depth_image_ocv)
            count_save += 1            
            
            cv2.imshow("Image", image_ocv)
            cv2.imshow("Depth", depth_image_ocv)
            key = cv2.waitKey(10)
            """
            #################################
            # m_zedcvHandle.process_key_event(zed, key)
        else:
            sl.c_sleep_ms(1)
            # key = cv2.waitKey(1)
    plt.close('all')
    cv2.destroyAllWindows()
    m_CLanParams.camera.close()
    print("\nFINISH")

# def switch_pipline(stcher, img):
   
#     switcher = {
#         0:  SimpfiedRGBPiplines(img),            
#         1:  RGBPiplines(img),
#         2:  DepthMapPiplines(img)      
       
#     }
# def SimpfiedRGBPiplines(img):   
#     plt.figure('ZED Camera Live')
#     plt.title('Visible and Depth')        
#     plt.imshow(img,cmap='gray')
#     plt.show(block=False)
#     plt.pause(0.25)
#     plt.close()
#     m_CLneDTrcks.MainDeteTrack(img)

# def RGBPiplines(img):   
#     m_CFRGBLneTrcks.MainDeteTrack(img) 

def get_parser():
    
    parser =ArgParse()
    parser.add_argument('--numL', type=int, help="Initial Setting of total number of Lane to track !",default=4)
    parser.add_argument('--pipL', type=int, help="Detecting Pipe Line Choosing: simplified 1, complex- RGB (2) and Depth (3) !",default=2)

    return parser


if __name__ == "__main__":

    args = get_parser().parse_args()
    arwacmain(args)
