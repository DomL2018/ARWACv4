from __future__ import print_function
"""
recoding the resuls, and doing statistic analysis about offset from ground truth:
combine the average of error of 6 lanes, and calcualte the ovaral error
groundtruth is determined by the colorful - lankmark pluged in the filed
"""
# Python 2/3 compatibility
print(__doc__)
import sys
PY3 = sys.version_info[0] == 3
import cv2
import pyzed.sl as sl
# import pyclustering
# from pyclustering.cluster import xmeans
# import numpy as np
# from sklearn.cluster import MeanShift, estimate_bandwidth
# from sklearn.datasets.samples_generator import make_blobs
import matplotlib
matplotlib.use('TkAgg')


import matplotlib.pyplot as plt
import pylab
import math
import statistics
import argparse
import ntpath
import os, glob
# from moviepy.editor import VideoFileClip
from tools.KFcvFeatlDbLaneDeteTrackGpsSeg import KFcvFeatlDbLaneDeteTracks

# from filters.ParticleFilter import ParticleFilter
# built-in modules
import itertools as it
from itertools import count
from collections import OrderedDict
import cv2
import threading,time


from points_extract.keypointsextract import CKeyptsBlobs
from tools.CSysParams import CLanesParams,CAlgrthmParams,CCamsParams#,CYoloV3Params
# from CInitialVideoSettings_lbl import CVideoSrcSetting,CCameraParamsSetting,CRobtoParamSetting
# from tools.CInitialVideoSettings_lbl import CVideoSrcSetting,CCameraParamsSetting,CRobtoParamSetting
# from gt_metrics.Detector_GT import gt_detect
from tools.CInitialVideoSettingsLive import CVideoSrcSetting,CCameraParamsSetting#,CRobtoParamSetting

# from tools.HSVRGBSeg4PF import BuildHSVColorModel,HSVColorSegmt,RGBColorSegmt,HSVColorSegmtFull,RGBColorSegmtFull,PltShowing,cvSaveImages
from tools.svo_export import ParsingZED
# from tools.socketcom import CJetScketUDPClient #, CCtrlSocke
# from ARWACv2dom.mavlink_ctrl import CMavlinkCtrl


import enum

# m_MavlinkCtrl = object()
m_CLanParams = object()
m_CAgthmParams = object()
m_CamParams  = object()
# m_CLneDTrcks = object()
m_CKFFeatDbLneTrcks = object()
# m_CJetClnt = object()
# grasslanes = {} # dictionary to hold the lane divisions by lnae1 , 2, 3, ...
"""
lines = cv2.HoughLinesP(edges, 1, np.pi/180, 30, maxLineGap=250)
for line in lines:
    x1, y1, x2, y2 = line[0]
    cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
"""

"""grasstrack class
  each grasstrack is composed of a ROI, an ID and a Kalman filter
  so we create a grasstrack class to hold the object state
"""

class AppType(enum.Enum):
    LEFT_AND_RIGHT = 1
    LEFT_AND_DEPTH = 2
    LEFT_AND_DEPTH_16 = 3
def print_camera_information(cam):
    while True:
        res = input("Do you want to display camera information? [y/n]: ")
        if res == "y":
            print()
            print(repr((cam.get_self_calibration_state())))
            print("Distorsion factor of the right cam before calibration: {0}.".format(
                cam.get_camera_information().calibration_parameters_raw.right_cam.disto))
            print("Distorsion factor of the right cam after calibration: {0}.\n".format(
                cam.get_camera_information().calibration_parameters.right_cam.disto))

            print("Confidence threshold: {0}".format(cam.get_confidence_threshold()))
            print("Depth min and max range values: {0}, {1}".format(cam.get_depth_min_range_value(),
                                                                    cam.get_depth_max_range_value()))

            print("Resolution: {0}, {1}.".format(round(cam.get_resolution().width, 2), cam.get_resolution().height))
            print("Camera FPS: {0}".format(cam.get_camera_fps()))
            print("Frame count: {0}.\n".format(cam.get_svo_number_of_frames()))
            break
        elif res == "n":
            print("Camera information not displayed.\n")
            break
        else:
            print("Error, please enter [y/n].\n")

   

def main():
    """
    """

    global m_CLanParams
    global m_CAgthmParams
    global m_CamParams 
    # global m_CJetClnt
    global m_CKFFeatDbLneTrcks
    # m_CJetClnt = CJetScketUDPClient()    
    # global m_MavlinkCtrl

    # m_MavlinkCtrl = CMavlinkCtrl()  # temporitly for testing
    # 
    m_CLanParams=CLanesParams() 
    # YoloSetParams = CYoloV3Params() 
    # m_detector = CKeyptsBlobs()
    m_VideoSrcSet = CVideoSrcSetting()
    #topdown
    ###################################################################################
    m_CLanParams.issvo = True
    m_CLanParams.isavi = False
    #topdown
    ###################################################################################
    if m_CLanParams.issvo == True or  m_CLanParams.isavi == True:
        src_vid = m_VideoSrcSet.data07032021GeN1_com()
    else:
        src_vid = m_VideoSrcSet.data_live_setting()
    ###########################################################################
    # Pth_sve4paper='/media/dom/Elements/data/data_processing/Hirst_wht_pf/'
    # subpth = 'data09052021Asra_hm/'
    # basicname = 'pf0905_'

    m_CLanParams.campitch = src_vid['pitch']
    m_CLanParams.video_file_l =   src_vid['path']#="/media/dom/Elements/data/TX2_Risehm/28072020/1st/goVL720q.avi"
    m_CLanParams.LaneOriPoint = src_vid['LaneOriPointL']#=50
    m_CLanParams.LaneEndPoint = src_vid['LaneEndPointR']#=50
    m_CLanParams.lanebandwith= src_vid['lanebandwidth']#=80
   
    m_CLanParams.Numoflanes = src_vid['NumofLanes']#=6,5,4,3,

    m_CLanParams.laneoffset= src_vid['laneoffset']#=155
    m_CLanParams.lanegap=int((m_CLanParams.LaneEndPoint-m_CLanParams.LaneOriPoint)/(m_CLanParams.Numoflanes-1)) 
    if m_CLanParams.issvo == True or  m_CLanParams.isavi == True:
        print("Reading SVO file: {0}".format(m_CLanParams.video_file_l))
    
    print("lane gap : {0}".format(m_CLanParams.lanegap))     


    if m_CLanParams.Numoflanes==2:
        m_CLanParams.laneStartPo = int(m_CLanParams.lanegap*0.382)
    elif m_CLanParams.Numoflanes==3:
        m_CLanParams.laneStartPo = int(m_CLanParams.lanegap*0.5)
    # elif Numoflanes==4:
    #     laneStartPo = int(lanegap*0.618)
    # elif Numoflanes==5:
    #     laneStartPo = int(lanegap*0.809)
    else:
        m_CLanParams.laneStartPo = int(m_CLanParams.lanegap*0.618)

    if m_CLanParams.laneStartPo < m_CLanParams.lanebandwith:  # must keep in this way, other wise more lanes than setting up
        m_CLanParams.laneStartPo = m_CLanParams.lanebandwith 

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


    m_CLanParams.deltalane= m_CLanParams.lanegap    
    m_CLanParams.PixelSize=m_CLanParams.lanegap  * m_CLanParams.mtricsize  

    # Split filename and extension.    
    # head, tail = ntpath.split(video_fille_l)
    # (name_pref, name_ext) = os.path.splitext(tail)

    # m_CLanParams.upd_thrsh=10 # critirial for using particle filtering , updating started from 2nd , first for warming up
    # m_CLanParams.pf_start=0# control of updating 'id' with kalman filter >18: #
    """
    ###################################################################################  
    
    #####################################################################################
    """
    # Create ZED objects
    zed = sl.Camera()
    if zed.is_opened()==True:
        zed.close()


    # Specify SVO path parameter
    init_params = sl.InitParameters()
    # init_params.set_from_svo_file(str(video_fille_l))    
    init_params.camera_resolution = sl.RESOLUTION.HD720#.HD1080
    init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE
    init_params.coordinate_units = sl.UNIT.METER#MILLIMETER
    init_params.depth_minimum_distance=0.5#minimum depth perception distance to half meter
    init_params.depth_maximum_distance=20#maxium depth percetion distan to 20 meter
    init_params.camera_fps=30#set fps at 30
    
    if m_CLanParams.issvo ==True or m_CLanParams.isavi == True:
        init_params.set_from_svo_file(str(m_CLanParams.video_file_l))
        init_params.svo_real_time_mode = False  # Don't convert in realtime

    # Open ZED wiht specified parameter
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        # sys.stdout.write(repr(err))
        zed.close()

    m_CamParams = CCamsParams(zed, sl)

    # m_CLanParams.issvo = False
    # m_CLanParams.isavi = False

    
    m_CLanParams.deltalane= m_CLanParams.lanegap    
    m_CLanParams.PixelSize=m_CLanParams.lanegap * m_CLanParams.mtricsize 
   
    # Get image size
    m_CLanParams.image_size = zed.get_camera_information().camera_resolution
    # width = image_size.width
    # height = image_size.height
    # width_sbs = width * 2
    
    # Prepare side by side image container equivalent to CV_8UC4
    # svo_image_sbs_rgba = np.zeros((height, width_sbs, 4), dtype=np.uint8)

    # Prepare single image containers
    m_CamParams.left_image = sl.Mat()
    m_CamParams.right_image = sl.Mat()
    m_CamParams.depth_image = sl.Mat()
    m_CamParams.rt_param = sl.RuntimeParameters()
    m_CamParams.rt_param.sensing_mode = sl.SENSING_MODE.FILL


    # Start SVO conversion to AVI/SEQUENCE
    # sys.stdout.write("Converting SVO... Use Ctrl-C to interrupt conversion.\n")

    # nb_frames = zed.get_svo_number_of_frames()
    # print('total number of frames in .svo files := ', nb_frames) 
    ########################################
    ########################################
        
    # 创建主显示窗口
    # cv2.namedWindow("Tracking...")
    # 设置任性字典和firstFrame标志（该标志使得背景分割器能利用这些帧来构建历史）
    
    # firstFrame = True
    # frame_num = -1

    # vis_size = (avi_width,avi_height)
    # fourcc1 = cv2.VideoWriter_fourcc(*'XVID')
    # fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    # out1 = cv2.VideoWriter('./output_images/Wheatoutpy1.mp4',fourcc1, 20, (frame_width,frame_height))

    # svRes = '/media/dom/Elements/data/data_processing/data012012021svo/'    
    # svRctRes = Pth_sve4paper+'PfRes'+campitch+'tst.avi'
    # out2 = cv2.VideoWriter(svRctRes,fourcc, 10.0,vis_size)
    # out2 = cv2.VideoWriter('/media/dom/Elements/data/data_collection/Hirst_wheat_pf/12112020/WhtoutPf121120.avi',fourcc2, 20.0,vis_size)
    # print('outRes dimension: ', vis_size)   

    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # out = cv2.VideoWriter('./output_images/rhm_output.avi',fourcc, 20.0, (640,480))
    #一行一行地读取视频帧
    # m_CLanParams.mt,m_CLanParams.mt_inv = perspective_transform((w0,h0),laneoffset)
    video_ocv =   src_vid['path']#="/media/dom/Elements/data/TX2_Risehm/28072020/1st/goVL720q.avi"
    video_dpt =   src_vid['path_dpt']#="/media/dom/Elements/data/TX2_Risehm/28072020/1st/goVL720q.avi"
   
    outputpth =''#/home/dom/Documents/ARWAC/domReport/ref_images/Zedlivevideo.avi'
    m_CAgthmParams = CAlgrthmParams(zed,sl,video_ocv,video_dpt,outputpth)
    # m_CLanParams.FullRRGNumRowsInit = ArgAll.numL # initial lane settting of numbers for tracking
    # m_CAgthmParams.piplineSelet = ArgAll.pipL # Select different algorithm for detection lanes 1: for early tidy lane, 2: RGB highly adjoining lane, 3: Depth map on 2
    m_CAgthmParams.grasslanes={}
    # lane_regs={} # keep the lane's name holder , according to the lanes initialization at begining
    # output_as_video = False    
    # app_type =''# AppType.LEFT_AND_RIGHT
    m_CKFFeatDbLneTrcks=  KFcvFeatlDbLaneDeteTracks(m_CAgthmParams,m_CLanParams)
    m_CLanParams.mt,m_CLanParams.mt_inv = m_CKFFeatDbLneTrcks.perspective_transform((m_CLanParams.w0,m_CLanParams.h0),m_CLanParams.laneoffset)
    # m_CAgthmParams.m_detector = m_detector
    start_zed()

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
    global m_CKFFeatLneTrcks
    global m_CLanParams


    m_CamParams.image_ocv = None
    m_CamParams.frame_num = -1


    sim_out = 0.0
    last_out=sim_out
    sleepT = 0.05
    
    rb_sleepT = 0.0
    avg_value=0.0
    msg_scl=0.0
    msg_vec = list()

    scale = -10.0 # control the normalization
    ctrFlag = 1 # selection of methods
    Lbase=200.0 #cm
    Ibase = 240 # pixes : height detection windows size
    Wbase = Ibase**m_CLanParams.PixelSize # approximately , pixel is squared - detection windows height
    
    while m_CamParams.key != 113 : #F2: 
        
        start_time = time.time()
        if killed() == True:
            print('Terminated...!')
            break

         # using .svo files
        if m_CLanParams.issvo == True or m_CLanParams.isavi ==True:

            #############################
            if m_CamParams.cam.grab(m_CamParams.rt_param) != sl.ERROR_CODE.SUCCESS:
                print ('filed to extract frame in .svo file')
                # print(repr(status))
                break

            # svo_position = m_CamParams.cam.get_svo_position()
            m_CAgthmParams.frame_num=m_CAgthmParams.frame_num+1
            if m_CAgthmParams.frame_num<=m_CLanParams.frm2strart:
                continue

            # # Check if we have reached the end of the video
            # if svo_position >= (m_CamParams.totalNum_frame - 1):  # End of SVO
            #     sys.stdout.write("\nSVO end has been reached. Exiting now.\n")
            #     break 
            
            # if m_CamParams.frame_num%1!=0:
                # continue
            # Retrieve SVO images
            # zed.retrieve_image(left_image, sl.VIEW.LEFT)
            m_CamParams.cam.retrieve_image(m_CamParams.left_image, sl.VIEW.LEFT, sl.MEM.CPU,m_CamParams.image_size)
            orig_frame = m_CamParams.left_image.get_data()
            # Convert SVO image from RGBA to RGB
            orig_frame = cv2.cvtColor(orig_frame, cv2.COLOR_RGBA2RGB)
            orig_frame = cv2.cvtColor(orig_frame, cv2.COLOR_RGB2BGR)
            plt.figure('ZED SVO File')
            plt.title('Visible and Depth')        
            plt.imshow(orig_frame,cmap='gray')
            plt.show(block=True)
            plt.pause(0.05)
            plt.close()  

            m_CKFFeatDbLneTrcks.MainDeteTrack(orig_frame) 
            # msg['L'] = 555.0
            # msg['R'] = 999.0
            # msg['N'] = 0.0
            msgi = float(m_CLanParams.Rbt_Ofset)
           
            if ctrFlag == 1:
                # scaled by lane width
                msg_scl = scale*msgi/m_CLanParams.deltalane
                # print ('Robot control offset {inter} and scaled as {externa}'.format(inter = msgi, externa = msg_scl))                
                                
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


            msg_scl = round(msg_scl,3)

            # m_CJetClnt.Data2Send(msg_scl)
            print ('Steering Input {externa}'.format(externa = msg_scl))

            plt.pause(0.25)
            continue
        # using live video        
        err = m_CamParams.cam.grab(m_CamParams.runtime)
        if err == m_CamParams.sl.ERROR_CODE.SUCCESS:                      
            
            # m_CamParams.sl.c_sleep_ms(1)
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
            m_CamParams.cam.retrieve_image(m_CamParams.image_zed, m_CamParams.sl.VIEW.LEFT, m_CamParams.sl.MEM.CPU, m_CamParams.image_size)#(m_CamParams.new_width, m_CamParams.new_height))
            # m_CamParams.cam.retrieve_image(m_CamParams.depth_image_zed, m_CamParams.sl.VIEW.DEPTH, m_CamParams.sl.MEM.CPU, m_CamParams.image_size)#(m_CamParams.new_width, m_CamParams.new_height))
                        
            # To recover data from sl.Mat to use it with opencv, use the get_data() method
            # It returns a numpy array that can be used as a matrix with opencv
            m_CamParams.image_ocv = m_CamParams.image_zed.get_data()
            # m_CamParams.depth_image_ocv = m_CamParams.depth_image_zed.get_data()
            m_CAgthmParams.frame_num=m_CAgthmParams.frame_num+1
            if m_CAgthmParams.frame_num<10:#:=m_CLanParams.frm2strart:
                continue
            orig_frame = cv2.cvtColor(m_CamParams.image_ocv, cv2.COLOR_RGBA2RGB)
            orig_frame = cv2.cvtColor(orig_frame, cv2.COLOR_RGB2BGR)
        
            m_CKFFeatDbLneTrcks.MainDeteTrack(orig_frame)#,'','',m_CLanParams.pf_start,'') 


            msgi = float(m_CLanParams.Rbt_Ofset)  # lane offset convert to control signal
         
            if ctrFlag == 1:
                # scaled by lane width
                msg_scl = scale*msgi/m_CLanParams.deltalane
                # print ('Robot control offset {inter} and scaled as {externa}'.format(inter = msgi, externa = msg_scl))                
                                
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
            # m_MavlinkCtrl.steering_receive_camera(sim_out)  # temorily commented for testing
            last_out = sim_out 
            time.sleep(0.5)
            rb_sleepT = rb_sleepT+sleepT
            secs = math.floor((time.time() - start_time))       
            # print("--- %s seconds ---" % (time.time() - start_time))         
            print ('Live time Running: {seTime} seconds for Steering:  {externa}'.format(seTime =secs, externa = sim_out))

        else:
            m_CamParams.sl.c_sleep_ms(1)
                # key = cv2.waitKey(1)
    plt.close('all')
    cv2.destroyAllWindows()
    m_CamParams.cam.close()
    print("\nFINISH") 
             
  
if __name__ == "__main__":
    
  main()
  #analysis data
#   data_offset_analysis()


