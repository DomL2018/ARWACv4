#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import numpy as np
import pyzed.sl as sl
import cv2
from tools.zedcvstream import zedcv # import zed processing methods
from tools.CSysParams import CLanesParams,CAlgrthmParams,CCamsParams
import matplotlib.pyplot as plt
import threading
from tools.RGBDeteTrack_Smpl import CLneDetsTracks
# from tools.FullRGBLaclneDeteTrack import CFullRGBLneDetsTracks
from argparse import ArgumentParser as ArgParse
import os

# help_string = "[s] Save side by side image [d] Save Depth, [n] Change Depth format, [p] Save Point Cloud, [m] Change Point Cloud format, [q] Quit"
# prefix_point_cloud = "Cloud_"
# prefix_depth = "Depth_"
#path = "./"
path = "/home/dom/prototype/ZED/"
# count_save = 0
# mode_point_cloud = 0
# mode_depth = 0
# point_cloud_format = sl.POINT_CLOUD_FORMAT.POINT_CLOUD_FORMAT_XYZ_ASCII
# depth_format = sl.DEPTH_FORMAT.DEPTH_FORMAT_PNG
m_CLanParams = object()
m_CAgthmParams = object()
m_CamParams  = object()
m_CLneDTrcks = object()
# m_CFRGBLneTrcks = object()

def main(ArgAll):

    # global mode_depth
    # global mode_point_cloud
    # global count_save
    # global depth_format
    # global point_cloud_format
    # Create a ZED camera object
    

    zed = sl.Camera()
    if zed.is_opened()==True:
        zed.close()
    # Set configuration parameters
    init = sl.InitParameters()
    # init.camera_resolution = sl.RESOLUTION.RESOLUTION_HD1080
    # init.depth_mode = sl.DEPTH_MODE.DEPTH_MODE_ULTRA#.DEPTH_MODE_PERFORMANCE
    # init.coordinate_units = sl.UNIT.UNIT_METER
    # if len(sys.argv) >= 2 :
    #     init.svo_input_filename = sys.argv[1]
    input_type = sl.InputType()
    if len(sys.argv) >= 2 :
        input_type.set_from_svo_file(sys.argv[1])
        
    init = sl.InitParameters(input_t=input_type)
    init.camera_resolution = sl.RESOLUTION.HD1080
    init.depth_mode = sl.DEPTH_MODE.PERFORMANCE
    init.coordinate_units = sl.UNIT.METER#MILLIMETER
    init.depth_minimum_distance=0.5#minimum depth perception distance to half meter
    init.depth_maximum_distance=20#maxium depth percetion distan to 20 meter
    init.camera_fps=30#set fps at 30

    # Open the camera
    zed.close()
    err = zed.open(init)
    if err != sl.ERROR_CODE.SUCCESS :
        print(repr(err))
        zed.close()
        exit(1)

    # Display help in console
    # print_help()
    # numofsettings = arrset
    # print('Number of lane setting = \n', ArgSets.numL)
    # Set runtime parameters after opening the camera
    # runtime = sl.RuntimeParameters()
    # runtime.sensing_mode = sl.SENSING_MODE.SENSING_MODE_STANDARD

    # # Prepare new image size to retrieve half-resolution images
    # image_size = zed.get_resolution()
    # new_width = image_size.width /2
    # new_height = image_size.height /2

    # # Declare your sl.Mat matrices
    # image_zed = sl.Mat(new_width, new_height, sl.MAT_TYPE.MAT_TYPE_8U_C4)  # this for display view
    # depth_image_zed = sl.Mat(new_width, new_height, sl.MAT_TYPE.MAT_TYPE_8U_C4) # this for the display view
    
    # point_cloud = sl.Mat()

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
    global m_CLanParams
    global m_CAgthmParams
    global m_CamParams 
    global m_CLneDTrcks
    # global m_CFRGBLneTrcks
  
    m_CLanParams = CLanesParams()
    m_CamParams = CCamsParams(zed, sl)
    # inputpth ="/home/chfox/ARWAC/Essa-0/20190206_143625.mp4"  
    video_ocv = "/home/dom/ARWAC/data/Input/20190206_143625.mp4"
    video_dpt = ''    
    outputpth ='/home/dom/Documents/ARWAC/domReport/ref_images/Zedlivevideo.avi'

    m_CAgthmParams = CAlgrthmParams(zed,sl,video_ocv,video_dpt,outputpth)


    m_CLanParams.FullRRGNumRowsInit = ArgAll.numL # initial lane settting of numbers for tracking
    m_CAgthmParams.piplineSelet = ArgAll.pipL # Select different algorithm for detection lanes 1: for early tidy lane, 2: RGB highly adjoining lane, 3: Depth map on 2

    print ('Num of Lanes Setting = ', ArgAll.numL)
    #Algorithms encapsulatd here:
    m_CLneDTrcks = CLneDetsTracks(m_CAgthmParams,m_CLanParams) 
    # m_CFRGBLneTrcks = CFullRGBLneDetsTracks(m_CAgthmParams,m_CLanParams) 


    m_zedcvHandle = zedcv(zed,sl,path) # instance of ZED stream
    # start_zed(zed, runtime, camera_pose, viewer, py_translation,depth_zed,image_zed,depth_image_zed,new_width,new_height,point_cloud,key)
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
    global m_CLneDTrcks
    global m_CAgthmParams
    # global m_CFRGBLneTrcks

     #open camera for streaming , temporarily for the test on video files collected
    m_CLneDTrcks.CameraStreamfromfile()
    # m_CFRGBLneTrcks.CameraStreamfromfile()
    # m_CLneDTrcks.MainDeteTrack()#("/home/chfox/ARWAC/Essa-0/20190206_143625.mp4")
   
    while m_CamParams.key != 113 : #F2
        if killed() == True:
            print('Terminated...!')
            break
        err = m_CamParams.cam.grab(m_CamParams.runtime)
        if err == m_CamParams.sl.ERROR_CODE.SUCCESS :
                               
            m_CamParams.cam.retrieve_measure(m_CamParams.depth_zed, m_CamParams.sl.MEASURE.DEPTH) # application purpose
            # Load depth data into a numpy array
            m_CamParams.depth_ocv = m_CamParams.depth_zed.get_data()
            # Print the depth value at the center of the image
            print('center of image = ', m_CamParams.depth_ocv[int(len(m_CamParams.depth_ocv)/2)][int(len(m_CamParams.depth_ocv[0])/2)])

            m_CamParams.cam.retrieve_image(m_CamParams.image_zed, m_CamParams.sl.VIEW.LEFT) # Left image 
            # Use get_data() to get the numpy array === full
            m_CamParams.image_ocv = m_CamParams.image_zed.get_data()
            # report = np.hstack((image_ocv,depth_zed)) #stacking images side-by-side
            # Display the left image from the numpy array
            # cv2.imshow("Image", image_ocv)
            plt.figure('ZED Camera Live')
            plt.title('depth_zed')        
            plt.imshow(m_CamParams.depth_ocv,cmap='gray')
            plt.show(block=False)
            plt.pause(0.5)
            plt.close()
            # Retrieve the left image, depth image in the half-resolution ----------- only for the display purpose
            # m_CamParams.cam.retrieve_image(m_CamParams.image_zed, m_CamParams.sl.VIEW.VIEW_LEFT, m_CamParams.sl.MEM.MEM_CPU, int(m_CamParams.new_width), int(m_CamParams.new_height))
            # m_CamParams.cam.retrieve_image(m_CamParams.depth_image_zed, m_CamParams.sl.VIEW.VIEW_DEPTH, m_CamParams.sl.MEM.MEM_CPU, int(m_CamParams.new_width), int(m_CamParams.new_height))
            m_CamParams.cam.retrieve_image(m_CamParams.image_zed, m_CamParams.sl.VIEW.LEFT, m_CamParams.sl.MEM.CPU, m_CamParams.image_size)#(m_CamParams.new_width, m_CamParams.new_height))
            m_CamParams.cam.retrieve_image(m_CamParams.depth_image_zed, m_CamParams.sl.VIEW.DEPTH, m_CamParams.sl.MEM.CPU, m_CamParams.image_size)#(m_CamParams.new_width, m_CamParams.new_height))
                       
            # To recover data from sl.Mat to use it with opencv, use the get_data() method
            # It returns a numpy array that can be used as a matrix with opencv
            m_CamParams.image_ocv = m_CamParams.image_zed.get_data()
            m_CamParams.depth_image_ocv = m_CamParams.depth_image_zed.get_data()

            # report = np.hstack((image_ocv,depth_ocv)) #stacking images side-by-side
            report = np.hstack((m_CamParams.image_ocv,m_CamParams.depth_image_ocv)) #stacking images side-by-side
            # cv2.imwrite('/home/chfox/Documents/ARWAC/FarmLanedetection/ref_images/Fig18EdgsDepth.jpg',regionofintrest)
            # cv2.imwrite('/home/chfox/Documents/ARWAC/FarmLanedetection/ref_images/Fig18HoughLineDepth.jpg',masked)
            # cv2.imwrite('/home/chfox/Documents/ARWAC/FarmLanedetection/ref_images/Fig18EdgeandHhLinesDepth.jpg',report)
            plt.figure('ZED Camera Live')
            plt.title('Visible and Depth')        
            plt.imshow(report,cmap='gray')
            plt.show(block=False)
            plt.pause(0.5)
            plt.close()
            # if m_CAgthmParams.piplineSelet==1:
            SimpfiedRGBPiplines(m_CamParams.image_ocv)
            #     pass
            # elif m_CAgthmParams.piplineSelet==2:
                # RGBPiplines(m_CamParams.image_ocv)     
                # pass          
            # RGBPiplines(m_CamParams.image_ocv)    
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
            m_CamParams.sl.c_sleep_ms(1)
            # key = cv2.waitKey(1)
    plt.close('all')
    cv2.destroyAllWindows()
    m_CamParams.cam.close()
    print("\nFINISH")

# def switch_pipline(stcher, img):
   
#     switcher = {
#         0:  SimpfiedRGBPiplines(img),            
#         1:  RGBPiplines(img),
#         2:  DepthMapPiplines(img)      
       
#     }
def SimpfiedRGBPiplines(img):   
    # plt.figure('ZED Camera Live')
    # plt.title('Visible and Depth')        
    # plt.imshow(img,cmap='gray')
    # plt.show(block=False)
    # plt.pause(0.25)
    # plt.close()
    m_CLneDTrcks.MainDeteTrack(img)

# def RGBPiplines(img):   
#     # plt.figure('ZED Camera Live')
#     # plt.title('Visible and Depth')        
#     # plt.imshow(img,cmap='gray')
#     # plt.show(block=False)
#     # plt.pause(0.25)
#     # plt.close()
#     m_CFRGBLneTrcks.MainDeteTrack(img)
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
    parser.add_argument('--numL', type=int, help="Initial Setting of total number of Lane to track !",default=6)
    parser.add_argument('--pipL', type=int, help="Detecting Pipe Line Choosing: simplified 1, complex- RGB (2) and Depth (3) !",default=1)

    return parser


if __name__ == "__main__":

    args = get_parser().parse_args()
    main(args)