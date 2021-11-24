#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pickle import TRUE
import sys
import numpy as np
import pyzed.sl as sl
import cv2
import matplotlib
# matplotlib.use('TkAgg')
# from tools.zedcvstream import zedcv # import zed processing methods
# from tools.CSysParams import CLanesParams,CAlgrthmParams,CCamsParams
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import threading
# from tools.LaneDeteTrack import CLneDetsTracks
from tools.FullRGBLaneDeteTrack import CFullRGBLneDetsTracks
# from tools.FullRGBLaneDeteTrack_mix import CFullRGBLneDetsTracks
from argparse import ArgumentParser as ArgParse
import os
from tools.CSysParamsAll import CLanesParams #,CAlgrthmParams,CCamsParams#,CYoloV3Params
from tools.CInitialVideoSettings import CVideoSrcSetting#,CCameraParamsSetting,CRobtoParamSetting
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
# m_CAgthmParams = object()
# m_CamParams  = object()
# m_CLneDTrcks = object()
m_CFRGBLneTrcks = object()
m_CJetClnt = object()
# keep roecording the data being sent out.......
from csv import writer
recdpth = '/home/dom/ARWAC/CropJetsnLanGuiCom/output/ctrldatsent.csv'
def append_list_as_row(file_name, list_of_elem):
    # Open file in append mode
    with open(recdpth, 'a+', newline='') as write_obj:
        # Create a writer object from csv module
        csv_writer = writer(write_obj)
        # Add contents of list as last row in the csv file
        csv_writer.writerow(list_of_elem)

def arwacmain(ArgAll):

    global m_CLanParams
    # global m_CAgthmParams
    # global m_CamParams 
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
    src_vid = m_VideoSrcSet.data07032021GeN1_com()
    m_CLanParams.issvo = True
    m_CLanParams.isavi = False

    m_CLanParams.campitch = src_vid['pitch']
    m_CLanParams.video_file_l =   src_vid['path']#="/media/dom/Elements/data/TX2_Risehm/28072020/1st/goVL720q.avi"
    print("Reading SVO file: {0}".format(m_CLanParams.video_file_l))
    
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
    ############################################################
    m_CLanParams.camera = sl.Camera()
    if m_CLanParams.camera.is_opened()==True:
        m_CLanParams.camera.close()
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

    if m_CLanParams.issvo == True:
        init.set_from_svo_file(str(m_CLanParams.video_file_l))
        init.svo_real_time_mode = False  # Don't convert in realtime

    # Open the camera
    err = m_CLanParams.camera.open(init)
    if err != sl.ERROR_CODE.SUCCESS :
        print(repr(err))
        m_CLanParams.camera.close()
        exit(1)

   # Get image size
    m_CLanParams.image_size = m_CLanParams.camera.get_camera_information().camera_resolution
    # width = image_size.width
    # height = image_size.height
    # width_sbs = width * 2
    # m_CamParams = CCamsParams(zed, sl)
    # Prepare side by side image container equivalent to CV_8UC4
    # svo_image_sbs_rgba = np.zeros((height, width_sbs, 4), dtype=np.uint8)

    # Prepare single image containers
    # m_CamParams.image_size = image_size
    m_CLanParams.left_image = sl.Mat()
    m_CLanParams.right_image = sl.Mat()
    m_CLanParams.depth_image = sl.Mat()
    m_CLanParams.rt_param = sl.RuntimeParameters()
    m_CLanParams.rt_param.sensing_mode = sl.SENSING_MODE.FILL

    # Start SVO conversion to AVI/SEQUENCE
    sys.stdout.write("Converting SVO... Use Ctrl-C to interrupt conversion.\n")

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

    
    m_CLanParams.totalNum_frame = m_CLanParams.camera.get_svo_number_of_frames()
    print('total number of frames in .svo files := ', m_CLanParams.totalNum_frame) 
    # m_CLanParams.FullRGBNumRowsInit = Numoflanes
    video_ocv =   src_vid['path']#="/media/dom/Elements/data/TX2_Risehm/28072020/1st/goVL720q.avi"
    video_dpt =   src_vid['path_dpt']#="/media/dom/Elements/data/TX2_Risehm/28072020/1st/goVL720q.avi"
   
    outputpth ='/home/dom/Documents/ARWAC/domReport/ref_images/Zedlivevideo.avi'
    # m_CAgthmParams = CAlgrthmParams(zed,sl,video_ocv,video_dpt,outputpth)
    # m_CLanParams.FullRRGNumRowsInit = ArgAll.numL # initial lane settting of numbers for tracking
    # m_CAgthmParams.piplineSelet = ArgAll.pipL # Select different algorithm for detection lanes 1: for early tidy lane, 2: RGB highly adjoining lane, 3: Depth map on 2
    # print ('Num of Lanes Setting = ', ArgAll.numL)
    #Algorithms encapsulatd here:
    # m_CLneDTrcks = CLneDetsTracks(m_CAgthmParams,m_CLanParams) 
    m_CFRGBLneTrcks = CFullRGBLneDetsTracks(m_CLanParams)
    # m_zedcvHandle = zedcv(m_CLanParams.camera,sl,m_CLanParams.video_file_l) # instance of ZED stream
    # start_zed(zed, runtime, camera_pose, viewer, py_translation,depth_zed,image_zed,depth_image_zed,new_width,new_height,point_cloud,key)
    m_CLanParams.mt,m_CLanParams.mt_inv = m_CFRGBLneTrcks.perspective_transform((m_CLanParams.w0,m_CLanParams.h0),m_CLanParams.laneoffset)

    start_zed()
    
    

# def start_zed(m_CamParams):
#     zed_callback = threading.Thread(target=run, args=(zed, runtime, camera_pose, viewer, py_translation,depth_zed,image_zed,depth_image_zed,new_width,new_height,point_cloud,key))
#     zed_callback.start()

def start_zed():
    # global m_CamParams
     # zed_callback = threading.Thread(target=run, args=(m_CamParams.cam, m_CamParams.sl))
    stop_threads = False
    zed_callback = threading.Thread(target=run,args =(lambda : stop_threads,)) # must be iterable so ','
    zed_callback.start()

# def run(cam, runtime, camera_pose, viewer, py_translation):
def run(killed):       
    # global m_CamParams
    # global m_CLneDTrcks
    # global m_CAgthmParams
    global m_CFRGBLneTrcks
    global m_CLanParams
     #open camera for streaming , temporarily for the test on video files collected
    # m_CLneDTrcks.CameraStreamfromfile()
    # m_CFRGBLneTrcks.CameraStreamfromfile() # for .avi files
    # m_CLneDTrcks.MainDeteTrack()#("/home/chfox/ARWAC/Essa-0/20190206_143625.mp4")

    m_CLanParams.image_ocv = None
    m_CLanParams.frame_num = 0
    msg = {}
    while m_CLanParams.key != 113 : #F2
        if killed == True:
            print('Terminated...!')
            break

        # using .svo files
        if m_CLanParams.issvo == True:

            #############################
            if m_CLanParams.camera.grab(m_CLanParams.rt_param) != sl.ERROR_CODE.SUCCESS:
                print ('filed to extract frame in .svo file')
                # print(repr(status))
                break

            svo_position = m_CLanParams.camera.get_svo_position()
            m_CLanParams.frame_num=m_CLanParams.frame_num+1
            if m_CLanParams.frame_num<=m_CLanParams.frm2strart:
                continue

            # Check if we have reached the end of the video
            if svo_position >= (m_CLanParams.totalNum_frame - 1):  # End of SVO
                sys.stdout.write("\nSVO end has been reached. Exiting now.\n")
                break 
            
            if m_CLanParams.frame_num%1!=0:
                continue
            # Retrieve SVO images
            # zed.retrieve_image(left_image, sl.VIEW.LEFT)
            m_CLanParams.filter_start = m_CLanParams.filter_start +1
            m_CLanParams.camera.retrieve_image(m_CLanParams.left_image, sl.VIEW.LEFT, sl.MEM.CPU,m_CLanParams.image_size)
            """
            if app_type == AppType.LEFT_AND_RIGHT:
                zed.retrieve_image(m_CamParams.right_image, sl.VIEW.RIGHT)
            elif app_type == AppType.LEFT_AND_DEPTH:
                zed.retrieve_image(m_CamParams.right_image, sl.VIEW.DEPTH)
            elif app_type == AppType.LEFT_AND_DEPTH_16:
                zed.retrieve_measure(m_CamParams.depth_image, sl.MEASURE.DEPTH)
            """
            # Generate file names
            filename = "left%s.png" % str(svo_position).zfill(6)
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
            plt.show(block=False)
            plt.pause(0.05)
            plt.close()           
            
            # RGBPiplines(orig_frame) 
            m_CFRGBLneTrcks.MainDeteTrack(orig_frame) 
            msgi = m_CLanParams.Rbt_Ofset 
            msgo = sum(m_CLanParams.Ctrl_Ofset.values())/(m_CLanParams.Numoflanes*m_CLanParams.deltalane)
            print ('Robot control offset {inter} and {externa}'.format(inter = msgi, externa = msgo))
            msgi= msgi*(-10.0)
            if abs(msgi)>1:
                msgi=0.9*msgi/(abs(msgi))

            m_CJetClnt.Data2Send(msgi)

            detec_data = [m_CLanParams.frame_num]
            detec_data.append(msgi)
            append_list_as_row('', detec_data)
            print('\n')
            print("Sending out : {0}".format(msgi))
            plt.pause(0.25)
            continue

        # using live video

        err = m_CLanParams.camera.grab(m_CLanParams.runtime)
        if err == m_CLanParams.sl.ERROR_CODE.SUCCESS :
                               
            m_CLanParams.camera.retrieve_measure(m_CLanParams.depth_zed, m_CLanParams.sl.MEASURE.DEPTH) # application purpose
            # Load depth data into a numpy array
            m_CLanParams.depth_ocv = m_CLanParams.depth_zed.get_data()
            # Print the depth value at the center of the image
            print('center of image = ', m_CLanParams.depth_ocv[int(len(m_CLanParams.depth_ocv)/2)][int(len(m_CLanParams.depth_ocv[0])/2)])

            m_CLanParams.camera.retrieve_image(m_CLanParams.image_zed, m_CLanParams.sl.VIEW.LEFT) # Left image 
            # Use get_data() to get the numpy array === full
            m_CLanParams.image_ocv = m_CLanParams.image_zed.get_data()
            # report = np.hstack((image_ocv,depth_zed)) #stacking images side-by-side
            # Display the left image from the numpy array
            # cv2.imshow("Image", image_ocv)
            plt.figure('ZED Camera Live')
            plt.title('depth_zed')        
            plt.imshow(m_CLanParams.depth_ocv,cmap='gray')
            plt.show(block=False)
            plt.pause(0.05)
            plt.close()
            # Retrieve the left image, depth image in the half-resolution ----------- only for the display purpose
            # m_CamParams.cam.retrieve_image(m_CamParams.image_zed, m_CamParams.sl.VIEW.VIEW_LEFT, m_CamParams.sl.MEM.MEM_CPU, int(m_CamParams.new_width), int(m_CamParams.new_height))
            # m_CamParams.cam.retrieve_image(m_CamParams.depth_image_zed, m_CamParams.sl.VIEW.VIEW_DEPTH, m_CamParams.sl.MEM.MEM_CPU, int(m_CamParams.new_width), int(m_CamParams.new_height))
            m_CLanParams.camera.retrieve_image(m_CLanParams.image_zed, m_CLanParams.sl.VIEW.LEFT, m_CLanParams.sl.MEM.CPU, m_CLanParams.image_size)#(m_CamParams.new_width, m_CamParams.new_height))
            m_CLanParams.camera.retrieve_image(m_CLanParams.depth_image_zed, m_CLanParams.sl.VIEW.DEPTH, m_CLanParams.sl.MEM.CPU, m_CLanParams.image_size)#(m_CamParams.new_width, m_CamParams.new_height))
                       
            # To recover data from sl.Mat to use it with opencv, use the get_data() method
            # It returns a numpy array that can be used as a matrix with opencv
            m_CLanParams.image_ocv = m_CLanParams.image_zed.get_data()
            m_CLanParams.depth_image_ocv = m_CLanParams.depth_image_zed.get_data()

            # report = np.hstack((image_ocv,depth_ocv)) #stacking images side-by-side
            report = np.hstack((m_CLanParams.image_ocv,m_CLanParams.depth_image_ocv)) #stacking images side-by-side
            # cv2.imwrite('/home/chfox/Documents/ARWAC/FarmLanedetection/ref_images/Fig18EdgsDepth.jpg',regionofintrest)
            # cv2.imwrite('/home/chfox/Documents/ARWAC/FarmLanedetection/ref_images/Fig18HoughLineDepth.jpg',masked)
            # cv2.imwrite('/home/chfox/Documents/ARWAC/FarmLanedetection/ref_images/Fig18EdgeandHhLinesDepth.jpg',report)
            plt.figure('ZED Camera Live')
            plt.title('Visible and Depth')        
            plt.imshow(report,cmap='gray')
            plt.show(block=False)
            plt.pause(0.05)
            plt.close()
            # if m_CAgthmParams.piplineSelet==1:
            #     SimpfiedRGBPiplines(m_CamParams.image_ocv)
            #     pass
            # elif m_CAgthmParams.piplineSelet==2:
                # RGBPiplines(m_CamParams.image_ocv)     
                # pass          
            # RGBPiplines(m_CamParams.image_ocv) 
            m_CFRGBLneTrcks.MainDeteTrack(m_CLanParams.image_ocv) 

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
            m_CLanParams.sl.c_sleep_ms(1)
            # key = cv2.waitKey(1)
    plt.close('all')
    cv2.destroyAllWindows()
    m_CLanParams.cam.close()
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
    # plt.figure('ZED Camera Live')
    # plt.title('Visible and Depth')        
    # plt.imshow(img,cmap='gray')
    # plt.show(block=False)
    # plt.pause(0.25)
    # plt.close()
    # m_CFRGBLneTrcks.MainDeteTrack(img) 
# def DepthMapPiplines(img):   
#     plt.figure('ZED Camera Live')
#     plt.title('Visible and Depth')        
#     plt.imshow(img,cmap='gray')
#     plt.show(block=False)
#     plt.pause(0.25)
#     plt.close()
#     # m_CLneDTrcks.MainDeteTrack(img)

def get_parser():
    
    parser =ArgParse()
    parser.add_argument('--numL', type=int, help="Initial Setting of total number of Lane to track !",default=4)
    parser.add_argument('--pipL', type=int, help="Detecting Pipe Line Choosing: simplified 1, complex- RGB (2) and Depth (3) !",default=2)

    return parser


if __name__ == "__main__":

    args = get_parser().parse_args()
    arwacmain(args)
