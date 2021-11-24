import os
from pickle import TRUE
from re import S
import sys
import numpy as np
from numpy import inf
# import pyzed.sl as sl
import cv2
import argparse
import pandas as pd

from moviepy.editor import VideoFileClip

def get_parent_dir(n=1):
    """ returns the n-th parent dicrectory of the current
    working directory """
    current_path = os.path.dirname(os.path.abspath(__file__))
    for k in range(n):
        current_path = os.path.dirname(current_path)
    return current_path

prj_pth = os.getcwd()
fil_pth = get_parent_dir(1)

src_path = os.path.join(get_parent_dir(1), "2_Training", "src")
utils_path = os.path.join(get_parent_dir(1), "Utils")

# yolo_path = os.path.join(prj_pth,"gt_metrics")
# yolo_src_path = os.path.join(yolo_path,"src")
# yolo_utils_path = os.path.join(yolo_path, "Utils")
tools_path = get_parent_dir(0)

# sys.path.append(yolo_path)
# sys.path.append(yolo_src_path)
# sys.path.append(yolo_utils_path)
sys.path.append(tools_path)

# from keras_yolo3.yolo import YOLO
from tools.configuration import Configuration
configuration = Configuration()
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
        
        self.campitch='TX2'  # setting on 'tidy ' image
        self.video_file_l = ''
        self.Numoflanes=4 #total number of rows covered within ROI
        self.laneoffset =45
        self.recording_counter=0 
        self.issvo = True
        self.LaneOriPoint = 0# for the lane intialization      
        self.Lane_GTstr = 'Current True Pos: ' # simulatie hose postion
        self.LaneEndPoint=1880

        self.lanebandwith = 50
        self.lanegap=100
        self.laneStartPo = 20    #origin point for tracking 
        self.upd_thrsh = 10
        self.pf_start = 0
        self.ori=None
        self.image_size = ''

        self.row0 = 25
        self.col0 = 102

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



        self.low_green = np.array([30, 0, 0])# for the purpose of putput info for dispimg
        self.high_green = np.array([100, 255, 255])# propective transform matrix    

        self.frm2strart=10   # prospecitve transform
        self.avi_width=25   # prospecitve transform
        self.avi_height=25#25 #15  #25   
         
        ### depth map
        self.FullDptRow0 = 0
        self.FullDptH0=375
        self.FullDptCol0 = 200
        self.FullDptW0=300
        self.dispimg_l='' # for the purpose of output processed results
        self.dispimg_r=''
        self.dispinfo =''# for the purpose of putput info for dispimg

        self.minLineLength = 2 # Hough tranform the segments' minimum lengh : incrasing by number of lines decreasing
        self.maxLineGap = 2 # Hough transform , the segemtns's manximum gap increasing by number lines decreasing
     
        self.minLineLength_dpt = 50 # Hough tranform the segments' minimum lengh : incrasing by number of lines decreasing
        self.maxLineGap_dpt = 6 # Hough transform , the segemtns's manximum gap increasing by number lines decreasing
        ##########################################
        ##########################################
        
        global configuration# = Configuration()
        configuration.load()

        self.FullRRGNumRowsInit=configuration.config['laneparams']['FullRRGNumRowsInit'] #4
        self.FullDTHNumRowsInit=configuration.config['laneparams']['FullDTHNumRowsInit'] #4 #total number of rows covered within ROI
        self.recording_counter=configuration.config['laneparams']['recording_counter'] #0 
        self.deltalane =configuration.config['laneparams']['deltalane'] # 1080# for the lane intialization
        
        self.SIMPNumoflaneInit =configuration.config['laneparams']['SIMPNumoflaneInit'] ##6 #total number of rows covered within ROI

        self.SIMPbandwith=configuration.config['laneparams']['SIMPbandwith'] #75  # setting on 'tidy ' image
        self.SIMPRGBrow0 = configuration.config['laneparams']['SIMPRGBrow0'] #20
        self.SIMPRGBh0=configuration.config['laneparams']['SIMPRGBh0'] #1880
        self.SIMPRGBcol0 = configuration.config['laneparams']['SIMPRGBcol0'] #50
        self.SIMPRGBw0=configuration.config['laneparams']['SIMPRGBw0'] #1000
        self.SIMPLaneOriPoint=configuration.config['laneparams']['SIMPLaneOriPoint'] #5#origin point for tracking 

        self.FullRGBbandwith=configuration.config['laneparams']['FullRGBbandwith'] #30
        self.FullDptbandwith=configuration.config['laneparams']['FullDptbandwith'] #30

        self.SIMPOffset=configuration.config['laneparams']['SIMPOffset'] #75   # prospecitve transform
        self.FullRGBoffset=configuration.config['laneparams']['FullRGBoffset'] #25   # prospecitve transform
        self.FullDEPoffset=configuration.config['laneparams']['FullDEPoffset'] #25#25 #15  #25 
        self.FullRGBLaneOriPoint = configuration.config['laneparams']['FullRGBLaneOriPoint'] #25 # where the row started
        self.FullDEPLaneOriPoint = configuration.config['laneparams']['FullDEPLaneOriPoint'] #25 # where the row started


        self.FullRGBrow0 = configuration.config['laneparams']['FullRGBrow0'] #0
        self.FullRGBh0=configuration.config['laneparams']['FullRGBh0'] #375
        self.FullRGBcol0 = configuration.config['laneparams']['FullRGBcol0'] #200
        self.FullRGBw0=configuration.config['laneparams']['FullRGBw0'] #300

        self.FullDepthrow0 = configuration.config['laneparams']['FullDepthrow0'] #0
        self.FullDepthh0=configuration.config['laneparams']['FullDepthh0'] #375
        self.FullDepthcol0 = configuration.config['laneparams']['FullDepthcol0'] #200
        self.FullDepthw0=configuration.config['laneparams']['FullDepthw0'] 


        self.mt ='' # propective transform matrix  
        self.mt_inv = '' # prospective transform matix - inverse
        ########################################################
        self.PixelSize=configuration.config['laneparams']['PixelSize'] #mm        
        self.LaneGap=configuration.config['laneparams']['LaneGap'] #mm
        self.Ctrl_Ofset={}# hold the output of offset for each indiviudal lane in metrics of mm
        self.Lane_GTs={}# hold the ground truth of each indiviudal lane in metrics of mm or pixles
        self.Lane_GTs_Prv={}
        self.Detec_Err = {}
        self.Lane_Ofset_Total=list()# hold the all offsets as whole so to do later analysis
        self.Lane_GTs_keys=list() # hold the lane key for the groudtruth records
        self.Rbt_Ofset=0.0
        self.mtricsize = 30 #cm the pysical size of lane gap : 12.,15,18, 30cm
        #########################################################
        self.pitch = 0
        self.image_size=''       
        # self.LaneOriPoint=0 # for the purpose of output processed results
        # self.LaneEndPoint=300
        # self.lanebandwith=80
        self.crped_h0 =400# for the purpose of putput info for 
        self.crped_w0 =400# for the purpose of putput info for dispimg
        self.crped_col0 =400# for the purpose of putput info for dispimg
    

       
        
        # https://www.learnopencv.com/invisibility-cloak-using-color-detection-and-segmentation-with-opencv/#:~:text=The%20Hue%20values%20are%20actually,detection%20of%20skin%20as%20red.
        """
        The inRange function simply returns a binary mask, where white pixels (255) represent pixels that fall into the upper and lower limit range and black pixels (0) do not.
        The Hue values are actually distributed over a circle (range between 0-360 degrees) but in OpenCV to fit into 8bit value the range is from 0-180. The red color is represented by 0-30 as well as 150-180 values.
        We use the range 0-10 and 170-180 to avoid detection of skin as red. High range of 120-255 for saturation is used because our cloth should be of highly saturated red color. The lower range of value is 70 so that we can detect red color in the wrinkles of the cloth as well.
        mask1 = mask1 + mask2
        Using the above line, we combine masks generated for both the red color range. It is basically doing an OR operation pixel-wise. It is a simple example of operator overloading of +.
        Now that you understood how color detection is done you can change the H-S-V range and use some other mono-color cloth in place of red color. In fact, a green cloth would work better than a red one because green is farthest away from the human skin tone.
        """
        # # Range for lower red
        # self.low_red_lowrange = np.array([0,120,70])# 170 for the purpose of putput info for dispimg
        # self.high_red_lowrange = np.array([10,255,255])# propective transform matrix  
        # # Range for upper range
        # self.low_red_uprange = np.array([170,120,70])# 170 for the purpose of putput info for dispimg
        # self.high_red_uprange = np.array([180,255,255])# propective transform matrix      

        #  # Range for lower red
        # self.low_red_lowrange = np.array([0,70,50])# 170 for the purpose of putput info for dispimg
        # self.high_red_lowrange = np.array([10,255,255])# propective transform matrix  
        # # Range for upper range
        # self.low_red_uprange = np.array([170,70,50])# 170 for the purpose of putput info for dispimg
        # self.high_red_uprange = np.array([180,255,255])# propective transform matrix      

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



class CAlgrthmParams(object):
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
        self.ToIntializeSetting = True
        self.filter_start=0 # counting for in
        self.upd_thrsh=18 # updating started for tracking
        self.grasslanes={}
        self.issvo = False
        self.isavi = False
        self.m_detector = object()
        
        self.fourcc1 = cv2.VideoWriter_fourcc(*'XVID')
        self.output_video_pth = outputpth#'/home/chfox/Documents/ARWAC/domReport/ref_images/Zedlivevideo.avi'
        # self.fourcc2 = cv2.VideoWriter_fourcc('M','J','P','G')
        # self.out = cv2.VideoWriter(self.output_video_pth,self.fourcc1, 10, (self.frame_width,self.frame_height))
        # self.out2 = cv2.VideoWriter('/home/dom/Documents/domReport/ref_images/OriAvi.mp4',fourcc2, 10, (frame_width,frame_height))
        self.piplineSelet = 4 # The algorithm selection : simpfliled with easy field 1, complex 2 : RGB and Depth map 3 

class CYoloV3Params(object):
    def __init__(self):

        self.src_path = yolo_src_path#'/home/dom/ARWAC/awarc_lanedetect_statis/gt_metrics/src'#os.path.join(get_parent_dir(0),"src")
        self.utils_path = yolo_utils_path#'/home/dom/ARWAC/awarc_lanedetect_statis/gt_metrics/Utils'#os.path.join(get_parent_dir(0), "Utils")
        self.data_folder =os.path.join(yolo_path, "Data") #'/home/dom/ARWAC/awarc_lanedetect_statis/gt_metrics/Data'
        self.image_folder = os.path.join(self.data_folder, "Source_Images")
        self.image_test_folder = os.path.join(self.image_folder, "Test_Images")

        self.detection_results_folder = os.path.join(self.image_folder, "Test_Image_Detection_Results")
        self.detection_results_file = os.path.join(self.detection_results_folder, "Detection_Results.csv")

        self.model_folder = os.path.join(self.data_folder, "Model_Weights")

        self.model_weights = os.path.join(self.model_folder, "trained_weights_final_gt.h5")
        self.model_classes = os.path.join(self.model_folder, "data_classes.txt")

        self.anchors_path = os.path.join(self.src_path, "keras_yolo3", "model_data", "yolo_anchors.txt")

        self.FLAGS = None
        self.parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
        self.ParserParams()

    def GetFileList(self,dirName, endings=[".jpg", ".jpeg", ".png", ".mp4",".avi"]):
        # create a list of file and sub directories
        # names in the given directory
        listOfFile = os.listdir(dirName)
        allFiles = list()
        # Make sure all file endings start with a '.'
        strtm = '.'
        for i, ending in enumerate(endings):
            if ending[0] != ".":
                strtm+=ending
                endings[i] = strtm#"." + ending
        # Iterate over all the entries
        for entry in listOfFile:
            # Create full path
            fullPath = os.path.join(dirName, entry)
            # If entry is a directory then get the list of files in this directory
            if os.path.isdir(fullPath):
                allFiles = allFiles + GetFileList(fullPath, endings)
            else:
                for ending in endings:
                    if entry.endswith(ending):
                        allFiles.append(fullPath)
        return allFiles

    def ParserParams(self):

            # Delete all default flags
        self.parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
        """
        Command line options
        """

        self.parser.add_argument(
            "--input_path",
            type=str,
            default=self.image_test_folder,
            help="Path to image/video directory. All subdirectories will be included. Default is "
            + self.image_test_folder,
        )

        self.parser.add_argument(
            "--output",
            type=str,
            default=self.detection_results_folder,
            help="Output path for detection results. Default is "
            + self.detection_results_folder,
        )

        self.parser.add_argument(
            "--no_save_img",
            default=False,
            action="store_true",
            help="Only save bounding box coordinates but do not save output images with annotated boxes. Default is False.",
        )

        self.parser.add_argument(
            "--file_types",
            "--names-list",
            nargs="*",
            default=[],
            help="Specify list of file types to include. Default is --file_types .jpg .jpeg .png .mp4",
        )

        self.parser.add_argument(
            "--yolo_model",
            type=str,
            dest="model_path",
            default=self.model_weights,
            help="Path to pre-trained weight files. Default is " + self.model_weights,
        )

        self.parser.add_argument(
            "--anchors",
            type=str,
            dest="anchors_path",
            default=self.anchors_path,
            help="Path to YOLO anchors. Default is " + self.anchors_path,
        )

        self.parser.add_argument(
            "--classes",
            type=str,
            dest="classes_path",
            default=self.model_classes,
            help="Path to YOLO class specifications. Default is " + self.model_classes,
        )

        self.parser.add_argument(
            "--gpu_num", type=int, default=1, help="Number of GPU to use. Default is 1"
        )

        self.parser.add_argument(
            "--confidence",
            type=float,
            dest="score",
            default=0.25,
            help="Threshold for YOLO object confidence score to show predictions. Default is 0.25.",
        )

        self.parser.add_argument(
            "--box_file",
            type=str,
            dest="box",
            default=self.detection_results_file,
            help="File to save bounding box results to. Default is "
            + self.detection_results_file,
        )

        self.parser.add_argument(
            "--postfix",
            type=str,
            dest="postfix",
            default="_gt",#"_gt_row",
            help='Specify the postfix for images with bounding boxes. Default is "_catface"',
        )

        self.FLAGS = self.parser.parse_args()
        self.save_img = not self.FLAGS.no_save_img
        # file_types = FLAGS.file_types

        # if file_types:
        #     input_paths = GetFileList(FLAGS.input_path, endings=file_types)
        # else:
        #     input_paths = GetFileList(FLAGS.input_path)
        self.input_paths = self.GetFileList(self.FLAGS.input_path)
        # Split images and videos
        img_endings = (".jpg", ".jpeg", ".png")
        vid_endings = (".mp4", ".mpeg", ".mpg", ".avi")
        self.input_image_paths = []
        self.input_video_paths = []
        for item in self.input_paths:
            if item.endswith(img_endings):
                self.input_image_paths.append(item)
            elif item.endswith(vid_endings):
                self.input_video_paths.append(item)

        self.output_path = self.FLAGS.output
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)


        # define YOLO detector
        self.yolo = YOLO(
            **{
                "model_path": self.FLAGS.model_path,
                "anchors_path": self.FLAGS.anchors_path,
                "classes_path": self.FLAGS.classes_path,
                "score": self.FLAGS.score,
                "gpu_num": self.FLAGS.gpu_num,
                "model_image_size": (416, 416),
            }
        )

        # Make a dataframe for the prediction outputs
        self.out_df = pd.DataFrame(
            columns=[
                "image",
                "image_path",
                "xmin",
                "ymin",
                "xmax",
                "ymax",
                "label",
                "confidence",
                "x_size",
                "y_size",
            ]
        )

        # labels to draw on images
        self.class_file = open(self.FLAGS.classes_path, "r")
        self.input_labels = [line.rstrip("\n") for line in self.class_file.readlines()]
        print("Found {} input labels: {} ...".format(len(self.input_labels), self.input_labels))

class CCamsParams (object):
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
        self.new_width = int(self.image_size.width)
        self.new_height =int(self.image_size.height)
        self.image_size.width = int(self.image_size.width)
        self.image_size.height =int(self.image_size.height)

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

        self.image_ocv = None
        self.frame_num = -1

    def __del__(self):
        if self.cam is not None:
            # self.cam.release()
            self.cam.close()
            self.cam = None
        if self.sl is not None:
            # self.sl.release()
            self.sl = None