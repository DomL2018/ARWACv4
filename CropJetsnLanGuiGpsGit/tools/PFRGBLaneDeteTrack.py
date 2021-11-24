from __future__ import print_function
from pickle import TRUE
print(__doc__)
import sys
PY3 = sys.version_info[0] == 3
import cv2
import numpy as np
# from sklearn.cluster import MeanShift, estimate_bandwidth
# from sklearn.datasets.samples_generator import make_blobs
import matplotlib.pyplot as plt


import math
# import statistics
# import argparse
from collections import OrderedDict
# import os, glob
# from moviepy.editor import VideoFileClip
# built-in modules
import itertools as it
from itertools import count
# from tools.LaneTrackingKF import grasstrack
from filters.ParticleFilter import ParticleFilter
from tools.CSysParams import CLanesParams,CAlgrthmParams,CCamsParams
from tools.HSVRGBSeg4PF import BuildHSVColorModel,HSVColorSegmt,RGBColorSegmt,HSVColorSegmtFull,RGBColorSegmtFull,PltShowing,cvSaveImages
# AgthmParams=object()
"""grasstrack class
  each grasstrack is composed of a ROI, an ID and a Kalman filter
  so we create a grasstrack class to hold the object state
"""
# import enum
# grasslanes = {} # dictionary to hold the lane divisions by lnae1 , 2, 3, ...

class grasstrack():
  _ids = count(1)  
#   numofids=0
 
  def __init__(self, id, frame, track_pt,track_window,lines_mid_vec,bandwith,lanegap):
    """init the pedestrian object with track window coordinates"""
    # set up the roi
    self.id = id
    #counting instances of a grasstrack
    self.numofids  = next(self._ids)
    print('total of ids = {0}, lane ID = {1}'.format(self.numofids,self.id))
    x,y,w,h = track_window
    self.track_window = track_window
    self.lines_mid_vec = lines_mid_vec
    self.bandwith=bandwith
    self.lanegap = lanegap
    self.width = int(frame.shape[1])
    self.height = int(frame.shape[0])
    self.num_pf = int (self.bandwith*self.height*0.618)
    self.pre_updated_id = id
    self.roi = frame[y-h:y, x-w:x]#cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2HSV)
    # roi_hist = cv2.calcHist([self.roi], [0], None, [16], [0, 180])
    # self.roi_hist = cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
    # set up the particle
    # self.xl_cent_pf=ParticleFilter(N=1000,x_range=(0,1500),sensor_err=1,par_std=100)
    self.band =self.bandwith #int (self.bandwith)# lanegap
    # self.xl_cent_pf=ParticleFilter(N=self.num_pf,x_range=(id-band,id+band),lane_id=self.id,sensor_err=1,par_std=bandwith)
    self.xl_cent_pf=ParticleFilter(N=self.num_pf,x_range=(self.id-self.band,self.id+self.band),lane_id=self.id,sensor_err=1,par_std=bandwith)

     #tracking queues
    self.xl_cent_q = [0]*15     
    self.count = 0
    # set up the kalman
    # self.kalman = cv2.KalmanFilter(4,2)
    # self.kalman.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]],np.float32)
    # self.kalman.transitionMatrix = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]],np.float32)
    # self.kalman.processNoiseCov = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],np.float32) * 0.05  #0.03
    self.measurement = np.array((2,1), np.float32) 
    self.prediction = np.zeros((2,1), np.float32)
    # self.term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )
    self.track_pt=track_pt
    self.center = None
    # self.update(frame,track_pt,track_window)
    
  def __del__(self):
    print ("lane %d destroyed" %self.id)

  def update(self, frame,track_pt,track_window):
    print("updating id = %d" %self.id)
    self.track_pt=track_pt
    self.track_window=track_window
    # print "updating %d " % self.id
    # 转换为HSV
    # hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # 计算行人HSV直方图的反向投影
    # back_project = cv2.calcBackProject([hsv],[0], self.roi_hist,[0,180],1)
    algorithm=1
    # 使用CAMShift来跟踪行人的运动，并根据行人的实际位置校正卡尔曼滤波器
    # if args.get("algorithm") == "c":
    if algorithm == 1:
    #   ret, self.track_window = cv2.CamShift(back_project, self.track_window, self.term_crit)
    #   pts = cv2.boxPoints(ret)
    #   ret=self.track_window
    #   pts = cv2.boxPoints(ret)
    #   pts = np.int0(pts)
    #   self.center = center(pts)
    #   cv2.polylines(frame,[pts],True, 255,1)
      x,y,w,h = self.track_window
      x=x-w
      y=y-h
      img2 = cv2.rectangle(frame, (x,y), (x+w,y+h), (168,0,0),2)
     
    # 使用均值漂移来跟踪行人的运动，并根据行人的实际位置校正卡尔曼滤波器
    # if not algorithm == 1:
    # # if not args.get("algorithm") or args.get("algorithm") == "m":    
    #   ret, self.track_window = cv2.meanShift(back_project, self.track_window, self.term_crit)
    #   x,y,w,h = self.track_window
    #   self.center = center([[x,y],[x+w, y],[x,y+h],[x+w, y+h]])  
    #   cv2.rectangle(frame, (x,y), (x+w, y+h), (255, 255, 0), 2)

    # lane_coords=[]
    # datacx=[]
    # datacy=[]
    # for point in self.lines_mid_vec: 
    #     if abs(self.id-point[0][0])<self.bandwith:
    #             cx=point[0][0]
    #             cy=point[0][1]
    #             datacx.append(cx)
    #             datacy.append(cy)
    #             lane_coords.append((cx, cy)) 

    # self.cx_median= statistics.median(datacx)
    # # print("sefl.cx_median",self.cx_median)
    # self.cy_median= statistics.median(datacy)
    # print("sefl.cy_median",self.cy_median)
    # self.center=np.median(lane_coords)      #update the center points , which will be used as the next update/correction of Kalman filter           
    # print('median center',self.center)
    # self.center=np.array([np.float32(self.cx_median), np.float32(self.cy_median)], np.float32)
    self.center=np.array([np.float32(self.track_pt[0]), np.float32(self.track_pt[1])], np.float32)

    # self.center=self.kalman.correct(self.center)
    self.track_pt[0] = self.xl_cent_pf.filterdata(self.track_pt[0])
    if abs(self.track_pt[0]-self.center[0])>self.band*0.618:
        self.track_pt = self.track_pt*0.382 + self.center*0.618
    else:  
        self.center = self.track_pt
        # self.center[1] = self.track_pt[1
    # print("state correction: x=%4.2f, and y= %4.2f "%(self.center[0], self.center[1]))
    tmx=self.center[0]
    tmy=self.center[1]

    if self.center[0]>self.width-4:
        tmx=self.width-4
    if self.center[1]>self.height-4:
        tmy=self.height-4

    cv2.circle(frame, (int(tmx), int(tmy)), 4, (0, 0,255), -1) 
    frame = cv2.line(frame,(int(tmx),self.height-1),(int(tmx),0),(255,255,0),4)

    # cv2.circle(frame, (int(self.center[0]), int(self.center[1])), 4, (0, 0,255), -1) 
    # frame = cv2.line(frame,(int(self.center[0]),height-1),(int(self.center[0]),0),(255,255,0),4)

    self.prediction = self.center#self.kalman.predict()      
    # print("state prediction:x=%4.2f, and y= %4.2f "%(self.prediction[0],self.prediction[1]))  
   
    return frame
class CPFRGBLneDetsTracks(CAlgrthmParams,CLanesParams):
    def __init__(self,AgthmParams,LanesParams):
        # global AgthmParams
        # AgthmParams = abb()
        self.AgthmParams = AgthmParams
        # self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.LanesParams = LanesParams        

    """
    def center(self,points):
        #calculates centroid of a given matrix
        x = (points[0][0] + points[1][0] + points[2][0] + points[3][0]) / 4
        y = (points[0][1] + points[1][1] + points[2][1] + points[3][1]) / 4
        return np.array([np.float32(x), np.float32(y)], np.float32)"""

    
    """    
    def get_dic(self,cent_refined,_lines_mid_v,masked,bandwith):      
        lane_dic = {}
        lane_coords = [] 
        for cen in cent_refined: 
            for point in _lines_mid_v: 
                if abs(cen-point[0][0])<bandwith:
                    cx=point[0][0]
                    cy=point[0][1]
                    lane_coords.append((cx, cy)) 
                    lkeys=(int)(cen)
                    if lkeys in lane_dic:
                        lane_dic[lkeys].append((cx, cy)) 
                    else:
                        lane_dic[lkeys]=[(cx,cy)]
                    # else: 
                    #     Output[y] = [(x, y)] 
            # print('lenth of refined lane_points = ',len(lane_coords),len(lane_dic))
            # print(lane_dic)
        lane_dic = OrderedDict(sorted(lane_dic.items()))
        return  lane_dic
    """
    def get_lanes(self,datax,datay,_lines_mid_v,masked,cen_hight,bandwith):   
    
        global grasslanes # lane division regeistration by names
        self.AgthmParams.grasslanes = OrderedDict(sorted(self.AgthmParams.grasslanes.items()))
        lanekeys = list(self.AgthmParams.grasslanes)
        keys_len = len(lanekeys)

        for k in range(keys_len):
            # id = grasslanes[lanekeys[k]].id
            _lines_mid_v=self.AgthmParams.grasslanes[lanekeys[k]].lines_mid_vec        

            datax=[row[0] for row in _lines_mid_v]
            datay=[row[1] for row in _lines_mid_v]
            # idrec = np.median (_lines_mid_v,axis=0)  # along  column
            # idrer = np.median (_lines_mid_v,axis=1)  # along row
            if len(datax)>0:
                id = np.median(datax)
                self.AgthmParams.grasslanes[lanekeys[k]].id = int((self.AgthmParams.grasslanes[lanekeys[k]].id+id)*0.5)

                
        return self.AgthmParams.grasslanes ,masked,bandwith
    #line segments detection for particles , 
    def get_lines(self,_lines_mid_v,masked):     

        # global grasslanes # lane division regeistration by names
        self.AgthmParams.grasslanes = OrderedDict(sorted(self.AgthmParams.grasslanes.items()))
        lanekeys = list(self.AgthmParams.grasslanes)
        keys_len = len(lanekeys)
        laneofcenter =0 # threshold of number of data storted for lane formation in _lines_mid_v
        for k in range(keys_len):#range(LanesParams.SIMPNumoflaneInit):
            # print('lane_centers before: ', AgthmParams.grasslanes[keys[k]].lane_centers)
            # AgthmParams.grasslanes[keys[k]].lane_centers.clear()
            # print('lane_centers after: ', AgthmParams.grasslanes[keys[k]].lane_centers)

            cens_lengh = len(self.AgthmParams.grasslanes[lanekeys[k]].lines_mid_vec)
            # print('track_pt: ', AgthmParams.grasslanes[keys[k]].lane_centers)

            if cens_lengh > laneofcenter:
                inx = cens_lengh-laneofcenter
                # AgthmParams.grasslanes[keys[k]].lane_centers = AgthmParams.grasslanes[keys[k]].lane_centers[inx:]
                self.AgthmParams.grasslanes[lanekeys[k]].lines_mid_vec =[]
            """     
            """
            lenth = len(_lines_mid_v)
            T = [x.pt for x in _lines_mid_v]
            datax=[row[0] for row in T]
            datay=[row[1] for row in T]        

            key_f = lambda x: x[0]
            for key, group in it.groupby(T, key_f):                    
                val =list(group)
                val=val.pop()
                for k in range(keys_len):
                    id = lanekeys[k] #lanekeys[k]#grasslanes[lanekeys[k]].id #
                    # lanewidth =  0.5*grasslanes[lanekeys[k]].bandwith
                    lanewidth =  0.382*self.AgthmParams.grasslanes[lanekeys[k]].lanegap
                    if key>id-lanewidth and key<id+lanewidth:
                        self.AgthmParams.grasslanes[lanekeys[k]].lines_mid_vec.append(val)
                        # print(str(key) + ': ' + str(val))
                        break

                    
            
            """
            # values = set(map(lambda x:x[1], T))
            # newT = [[y[0] for y in T if y[1]==x] for x in values]

            datax=[row[0] for row in T]
            datay=[row[1] for row in T]
            """
            widthx = len(datax)
            # heighty = len(datay)
            # print('widthx=', widthx)
            # print('heighty=', heighty)
            lenx = np.linspace(0, widthx, widthx)
            # leny = np.linspace(0, heighty, heighty)
            # pylab.scatter(lenx,datax)
            # fig = pylab.gcf()
            # fig.canvas.set_window_title('Tracking...')
            # plt.title('center of lane')
            # pylab.scatter(datax,datay)
            # # pylab.scatter(lenx,datax)
            # pylab.savefig("./output_images/lincenters_wheat.png")
            # plt.show(block=False)
            # plt.pause(0.25)
            # plt.close()
            # pylab.show()

            # plt.show(block=False)
            # plt.pause(1)
            # plt.close()
            #clustering
            return datax, datay,_lines_mid_v

    # calculate and merge multipple detection into regression 
    def get_dic_centring(self,gt_keys, gt_cents_v,bandwith):   
            
        lane_coords_x = [] 
        lane_coords_y = [] 
        lane_dic_cents = {}
        for cen in gt_keys: 
            for point in gt_cents_v: 
                if abs(cen-point.pt[0])<bandwith:
                    cx=point.pt[0]
                    cy=point.pt[1]
                    lane_coords_x.append(cx) 
                    lane_coords_y.append(cy)
                    meanx = sum(lane_coords_x)/len(lane_coords_x)
                    meany = sum(lane_coords_y)/len(lane_coords_y)
                    lkeys=(int)(cen)                
                    meanx = np.mean(lane_coords_x)
                    meany = np.mean(lane_coords_y)

                    # meanx = np.mean(lane_coords,0)
                    # meany = np.mean(lane_coords,1)
                    """
                    if lkeys in lane_dic:
                        lane_dic_gt[lkeys].append((cx, cy)) 
                    else:
                        lane_dic_gt[lkeys]=[(cx,cy)]
                    # else: 
                    #     Output[y] = [(x, y)] 
                    """
                    lane_dic_cents[lkeys]=[meanx,meany]
            # print('lenth of groun truth lane enteroids = ',len(lane_coords),len(lane_dic))
            # print(lane_dic)

        return  lane_dic_cents
    def get_dic(self,cent_refined,_lines_mid_v,masked,bandwith):   
        
        lane_dic = {}
        lane_coords = [] 
        for cen in cent_refined: 
            for point in _lines_mid_v: 
                if abs(cen-point[0][0])<bandwith:
                    cx=point[0][0]
                    cy=point[0][1]
                    lane_coords.append((cx, cy)) 
                    lkeys=(int)(cen)
                    if lkeys in lane_dic:
                        lane_dic[lkeys].append((cx, cy)) 
                    else:
                        lane_dic[lkeys]=[(cx,cy)]
                    # else: 
                    #     Output[y] = [(x, y)] 
            # print('lenth of refined lane_points = ',len(lane_coords),len(lane_dic))
            # print(lane_dic)
        lane_dic = OrderedDict(sorted(lane_dic.items()))
        return  lane_dic
        # return [l[0] for l in lines_in]


    def draw_display(self,masked,grasslanes,cen_hight,width):
        
        cols = int(masked.shape[1] )
        rows = int(masked.shape[0])
        for key, cont in grasslanes.items(): # try to get each column
            
            id = grasslanes[key].id
            coords=grasslanes[key].lines_mid_vec   
            
            lth=len(coords)
            # print('lenth and coords = :',lth, coords)
            cnt1=[]
            sumx=0.0
            sumy=0.0
            sumxy=0.0
            sumx2=0.0
            sumy2=0.0

            if (lth>0):            
                for pt in coords:      
                    cnt1.append([pt])
                    # print('pt = : ', pt)              
                

                    x=pt[0]
                    y=pt[1]
                    x2=x*x   #square of x
                    y2=y*y  # square of y
                    xy=x*y # inner x and y coordinates
                    sumx=sumx+x
                    sumy=sumy+y
                    sumxy=sumxy+xy
                    sumx2=sumx2+x2
                    sumy2=sumy2+y2
                
                # print('cnt1 = : ', cnt1)      
                cx=(int)(key)
                cxy= np.median(coords,axis=0)#cen_hight 
                cx=cxy[0]
                cy=cxy[1]
                # print('cy-mdian = ', cy[1])
                cnt1.append([(cx,cy)])
                cnt1=np.array(cnt1, np.int64)
                x=cx
                y=cy
                x2=x*x   #square of x
                y2=y*y  # square of y
                xy=x*y # inner x and y coordinates
                sumx=sumx+x
                sumy=sumy+y
                sumxy=sumxy+xy
                sumx2=sumx2+x2
                sumy2=sumy2+y2

                denominator=lth*sumx2 -sumx*sumx
                if denominator <0.0000001:
                    denominator = 0.0000001
                    print('denominator = : ',denominator)  

                a= (sumy*sumx2-sumx*sumxy)/denominator         #https://www.statisticshowto.datasciencecentral.com/probability-and-statistics/regression-analysis/find-a-linear-regression-equation/
                b= (lth*sumxy-sumx*sumy)/denominator

                # print('a = : ', a, 'b = : ', b)                           
                # cnt = np.array([[100,794],[124,415],[111,798],[95,935]], np.int32)               
                # print('cnt = : ', cnt)
                [vx,vy,x,y] = cv2.fitLine(cnt1, cv2.DIST_L2,0,0.01,0.01)
                lefty = int((-x*vy/vx) + y)
                righty = int(((cols-x)*vy/vx)+y)
                # print('lefty = ', lefty, 'righty = ', righty)
                if abs(lefty)>width or abs(righty)>width:
                # lefty =(width if (lefty>width) else 0)               
                    masked = cv2.line(masked,(x,rows-1),(x,0),(200,0,200),5)
                # lefty =(int)(a+b*(cols-1))     #y=a+bx
                # masked = cv2.line(masked,(cols-1,lefty),(cx,cy),(255,0,255),8)
                else:
                    # masked = cv2.line(masked,(cols-1,lefty),(0,righty),(200,255,200),8)
                    # masked = cv2.line(masked,(cols-1,righty),(0,lefty),(200,255,200),8)
                    masked = cv2.line(masked,(lefty,rows-1),(righty,0),(200,255,200),5)
            #draw_pathlines(masked, lane_dic, color=[255, 0, 0], thickness=8, make_copy=True)
        PltShowing(masked,'Tracking...','RGB & HSV', False, False)
        # cvSaveImages(masked,'im_with_lineReg_Pf.jpg')

        return masked 

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
        
    def CameraStreamfromfile(self):
        #/home/chfox/ARWAC/Essa-0/images    
        # video_file = '/home/chfox/ARWAC/KFPFTrack/dataset/soccer_02.mp4'
        print('ocv: ={0}'.format(self.AgthmParams.video_ocv))
        try:
            # to use a non-buffered camera stream (via a separate thread)
            if not(self.AgthmParams.video_ocv):
                import camera_stream
                self.AgthmParams.camera = camera_stream.CameraVideoStream()
            else:
                # Check if camera opened successfully
                self.AgthmParams.camera = cv2.VideoCapture(self.AgthmParams.video_ocv) 
                if (self.AgthmParams.camera.isOpened()== False): 
                    print("Error opening video stream or file")
                    # cap = cv2.VideoCapture() # not needed for video files
                else:
                    self.AgthmParams.totalNum_frame =int(self.AgthmParams.camera.get(cv2.CAP_PROP_FRAME_COUNT))
                    print('frame total numer={0}, width={1}, Height={2}'.format(self.AgthmParams.totalNum_frame,self.AgthmParams.frame_width,self.AgthmParams.frame_height))
                    self.AgthmParams.video_image = np.zerosk((self.AgthmParams.frame_height,self.AgthmParams.frame_width,3), np.uint8)
                    self.AgthmParams.video_counter = self.AgthmParams.totalNum_frame*2
                    
        except:
            # if not then just use OpenCV default
            print("INFO: camera_stream class not found - camera input may be buffered")
            self.AgthmParams.camera = cv2.VideoCapture(0)

    def MainDeteTrack(self,orig_frame):
        
        # print('Initial NumofRows Setting = {0}'.format(self.LanesParams.Numoflanes)) 
        ########################################################################
        self.AgthmParams.frame_num=self.AgthmParams.frame_num+1

        if self.LanesParams.isavi ==True:     
        
            if self.AgthmParams.frame_num == self.AgthmParams.totalNum_frame-1:
                # recording_counter=recording_counter+1
                self.AgthmParams.camera = cv2.VideoCapture(self.AgthmParams.video_ocv)
            grabbed, orig_frame = self.AgthmParams.camera.read()
            while (self.AgthmParams.frame_num<self.LanesParams.frm2strart):
                self.AgthmParams.frame_num+=1
                grabbed, orig_frame = self.AgthmParams.camera.read()            
                continue        

            if (self.AgthmParams.frame_num>self.AgthmParams.video_counter or grabbed is False):
                print ("failed to grab frame.")
                self.AgthmParams.out.release()
                cv2.destroyAllWindows()
                return     
        
        ########################################################################

        self.LanesParams.pf_start=self.LanesParams.pf_start+1
        print( "\n -------------------- FRAME %d --------------------" % self.AgthmParams.frame_num)   
 

        row0 =self.LanesParams.row0 #= 0
        h0=self.LanesParams.h0#=375
        col0 =self.LanesParams.col0# = 200
        w0=self.LanesParams.w0#=300
        PltShowing(orig_frame,'Camera Output...','Origin', True,False)

        snip = orig_frame[row0:row0+h0,col0:col0+w0]
        PltShowing(snip,'Camera Output...','snip', True,False)

        snpwidth = int(snip.shape[1])
        lanebandwith=self.LanesParams.lanebandwith    #30 #estimation or setting the initial lane wide
        Numoflanes = self.LanesParams.Numoflanes#.FullDTHNumRowsInit  #4
        laneoffset= self.LanesParams.laneoffset  #25#25 #15  #25 
        LaneOriPoint = self.LanesParams.LaneOriPoint   # 25 # where the row started

        laneStartPo = self.LanesParams.laneStartPo#LaneOriPoint + int (lanebandwith*0.5)
        lanegap=self.LanesParams.lanegap #int((snpwidth-Numoflanes*lanebandwith-LaneOriPoint*2)/(Numoflanes-1))
        # self.LanesParams.deltalane= self.LanesParams.deltalane #lanebandwith+lanegap
             
        img_size = (snip.shape[1], snip.shape[0])
        m, m_inv = (self.LanesParams.mt,self.LanesParams.mt_inv) #self.perspective_transform(snip,laneoffset)

        warped = cv2.warpPerspective(snip, m, img_size, flags=cv2.INTER_LINEAR) 
        PltShowing(warped,'warping...','warped', True, False)
        cen_hight=(int)(0.382*warped.shape[0])
        # print('warped dimension: ', warped.shape)       
        #################################                    
 
        width = int(warped.shape[1] )
        height = int(warped.shape[0])

  
        masked = np.copy(warped)#cv2.bitwise_and(snip, snip, mask=mask)
        detect = np.copy(warped) 
        # hsv_frame = cv2.cvtColor(snip, cv2.COLOR_BGR2HSV)
        """       
        """
        Pth_sve4paper = ''
        im_2gbr,green_wheat = RGBColorSegmtFull(warped,Pth_sve4paper,'tx2',self.LanesParams.pf_start,'')
        # green_wheat = HSVColorSegmtFull(warped,low_green,high_green,name_pref)
        PltShowing(green_wheat,'green_wheat','green_wheat', True, False)
        # im_2gbr_3d = np.stack((im_2gbr,)*3, axis=-1)
        # res = np.hstack((im_2gbr_3d,green_wheat)) #stacking images side-by-side


        # low_green = self.LanesParams.low_green #np.array([gclor_l, gSatur_l, gval_l])
        # high_green = self.LanesParams.high_green  #np.array([gclor_h, gSatur_h, gval_h])
        # regionofintrest = HSVFullEdgeProc(warped,low_green,high_green,'Tx2') 

        # datax,datay,_lines_mid_v=self.get_lines(regionofintrest,masked)
        # PltShowing(masked,'detect...','line segements',False,False)
        # #get lane sortd by finding the center of each lane
        # cent_refined,masked,bandwith=self.get_lanes(datax,datay,_lines_mid_v,masked,cen_hight,lanebandwith)      
        #get dictionary centroid <=> lane : (x,y)
        # lane_dic= self.get_dic(cent_refined,_lines_mid_v,masked,lanebandwith)  


        ####################################################   
        #get dictionary centroid <=> lane : (x,y)
        # lane_dic= self.get_dic(cent_refined,_lines_mid_v,masked,bandwith)    
        # masked=self.draw_display(masked,lane_dic,cen_hight,width)

        if self.AgthmParams.ToIntializeSetting is True:
            rows = int(masked.shape[0])
            for la in range(laneStartPo,snpwidth+1,self.LanesParams.deltalane):                 
                self.AgthmParams.video_image = cv2.line(masked,(la,rows-1),(la,0),(0,0,0),10)

            self.AgthmParams.ToIntializeSetting = False
            return        
        ###########################################

        # for pt in cent_refined:
        if self.AgthmParams.firstFrame is True:
            # deltalane= lanebandwith+lanegap
            deltalane= lanegap
            print('Lane Gap = {0}, Lane Width = {1}, Total Lane width = {2}, loop Deta = {3}'.format(lanegap,lanebandwith,snpwidth,deltalane))
            NumofSettingID=0
            # for la in range(LaneOriPoint,snpwidth,deltalane):  
            for la in range(laneStartPo,snpwidth-laneStartPo+1,lanegap):           
                track_pt=[la,cen_hight]
                _lines_mid_v=[]#[(la, cen_hight)]
                x,y,w,h=la,cen_hight,10,10
                self.AgthmParams.grasslanes[la]=grasstrack(la,masked,track_pt,(x,y,w,h),_lines_mid_v,lanebandwith,deltalane)
                index='lane_'+str(la)
                # lane_regis[index] = grasslanes[la] # just for the registration of identification convenient
                NumofSettingID=NumofSettingID+1
                #temporay testing
                # index='lane'+str(NumofSettingID)
                # self.LanesParams.Lane_GTs[index]=la
                # self.LanesParams.Lane_GTs_keys.append(la)
                # lanes_gt=get_dic_gt(gt_keys, blobCentrods,bandwith)

            print('total number of lanes = {0}'.format(NumofSettingID))
                # print('total number of lanes = {0}'.format(grasstrack.numofids))
            self.AgthmParams.firstFrame=False
            # blurred=closing
        feat_pts,feat_img =self.AgthmParams.m_detector.get_cvFASTKeypoints(green_wheat)

            # if self.AgthmParams.firstFrame is True:
            #     # deltalane= self.LanesParams.FullRGBbandwith+lanegap
            #     # deltalane= lanebandwith+lanegap
            #     # print('Lane Gap = {0}, Lane Width = {1}, Total Lane width = {2}, loop Deta = {3}'.format(lanegap,self.AgthmParams.FullRGBbandwith,snpwidth,deltalane))
            #     # print('Lane Gap = {0}, Lane Width = {1}, Total Lane width = {2}, loop Deta = {3}'.format(lanegap,lanebandwith,snpwidth,deltalane))
            #     print('Lane Gap = {0}, Lane Width = {1}, Total Lane width = {2}, loop Deta = {3}'.format(lanegap,lanebandwith,snpwidth,self.LanesParams.deltalane))
            #     NumofSettingID=0
            #     # for la in range(self.LanesParams.FullRGBLaneOriPoint,snpwidth,deltalane):  
            #     # for la in range(LaneOriPoint,snpwidth,deltalane):      
            #     for la in range(lstartpt,snpwidth,self.LanesParams.deltalane):                 
            #         track_pt=[la,cen_hight]
            #         _lines_mid_v=[(la, cen_hight)]
            #         self.AgthmParams.grasslanes[la]=grasstrack(la,masked,track_pt,(x,y,w,h),_lines_mid_v,lanebandwith)
            #         NumofSettingID=NumofSettingID+1
            #     print('total number of lanes tracked = {0}'.format(NumofSettingID))
            #     # print('total number of lanes = {0}'.format(grasstrack.numofids))
            #     self.AgthmParams.firstFrame=False


        # apply mask and on canny image on screen
        regionofintrest = feat_pts # edges #cv2.bitwise_and(edges, edges, mask=mask) # mask, mask0
        #get line segments from region of interest
        datax,datay,_lines_mid_v=self.get_lines(regionofintrest,masked)
        # grasslanes,masked,bandwith=get_lanes(datax,datay,_lines_mid_v,masked,cen_hight,lanebandwith)      
  
        #get lane sortd by finding the center of each lane
        cent_refined,masked,bandwith=self.get_lanes(datax,datay,[],masked,cen_hight,lanebandwith)      
        #get dictionary centroid <=> lane : (x,y)
        # lane_dic= get_dic(cent_refined,_lines_mid_v,masked,lanebandwith)    
        masked = self.draw_display(feat_img,self.AgthmParams.grasslanes,cen_hight,width)

        cent_refined = []
        for ikey, p in self.AgthmParams.grasslanes.items():
            cent_refined.append(p.id)


        # lane_dic= self.get_dic(cent_refined,_lines_mid_v,masked,lanebandwith)    
        # self.draw_display(masked,lane_dic,cen_hight,width)

        self.LanesParams.Lane_Detect_val={}        
        if self.AgthmParams.firstFrame is False:
            # print('\n Particle tracking points ...........')
            for ikey, p in self.AgthmParams.grasslanes.items():#.iteritems(): 
                found_flg=False # tracking the updated lanes  
                # keep last step id
                pre_id = p.id
                pre_predic_id = p.pre_updated_id
                coords=self.AgthmParams.grasslanes[ikey].lines_mid_vec
                # for key, coords in lane_dic.items(): # try to get each column
                # draw  ikey for comparasion as allocation
                masked = cv2.line(masked,(ikey,height-1),(ikey,0),(0,0,0),2)
                lth=len(coords)
                # if lth==0:
                #     continue
                # masked = cv2.line(masked,(int(p.id),height-1),(int(p.id),0),(165,42,42),2)
                lane_centers=[]

                # for key, coords in lane_dic.items(): # try to get each column
                #     lth=len(coords)
                lane_centers=[]
                # id=int(key)
                if (lth>0):            
                    for pt in coords:      
                        lane_centers.append([pt])
                        # print('pt = : ', pt)         
                # print('lane_centers = : ', lane_centers)      
                # cx=(int)(key)
                
                cxy =np.array([[p.id,cen_hight]])
                if lth==0:
                    cxy[0][0] = p.id
                    cxy[0][1] =cen_hight
                else:
                    cxy= np.mean(lane_centers,axis=0)#

                track_pt=np.array((cxy[0][0],cxy[0][1]),np.float32)
                # print('track_pt = : ', track_pcxy[0][0]
                x,y,w,h=(int)(cxy[0][0]),(int)(cxy[0][1]),10,10
                # p.id = cxy[0][0]
                if abs(ikey-p.id)<self.LanesParams.deltalane*0.618:#(lanebandwith*1.618):                       
                    masked = p.update(masked,track_pt,(x,y,w,h))                       
                    p.id=ikey*0.382+cxy[0][0]*0.618 # must keep here before later applying kalman filter
                    # p.id=key  # for the wide lane , trust detection
                    if self.LanesParams.pf_start>=self.LanesParams.upd_thrsh: #abs(p.id-p.center[0])<(lanebandwith*0.1545):
                        # p.id = ikey*0.382 + p.center[0][0]*0.618
                        # p.pre_updated_id = p.prediction[0][0]
                        p.id = ikey*0.382 + p.center[0]*0.618
                        p.pre_updated_id = p.prediction[0]
                    #    p.id = ikey*0.5 + p.center[0]*0.5
                    #    detect = p.update(detect,track_pt,(x,y,w,h))
                    xpos = int(p.id)
                    if xpos > width:
                        xpos = width-4
                    elif xpos <4:
                        xpos = 4
                    detect = cv2.line(detect,(xpos,height-1),(xpos,0),(255,255,0),4)                       
                    # lane_dic.pop(key)
                    found_flg=True
                    # break
            if found_flg==False:
                print("Lande id: %d is not updated!" % p.id)
                if self.LanesParams.pf_start>self.LanesParams.upd_thrsh:
                    pred_posx=p.id*0.618+pre_predic_id*0.382
                    # pred_posx=p.id*0.809+pre_predic_id*0.191
                else:
                    pred_posx=p.id
                    # if abs(ikey-pred_posx)<lanebandwith*1.618:#(int)(lanebandwith*0.382):
                    #     p.id=(pred_posx+ikey)*0.5
                p.id=ikey*0.618+pred_posx*0.382   # for the wider lane, the detectin is not a prolem , no ambugity, so the the detection has more weight or influence on
                # p.id=ikey*0.5+pred_posx*0.5
                    # lane_mat.masked = cv.line(lane_mat.masked,(pred_posx,height-1),(pred_posx,0),(255,255,0),4)
                    # masked = cv2.line(masked,(pred_posx,height-1),(pred_posx,0),(255,255,0),4)
                    # detect = cv2.line(detect,(pred_posx,height-1),(pred_posx,0),(255,255,0),4)
                    
                # else:
                #     # lane_mat.masked = cv.line(lane_mat.masked,(p.id,height-1),(p.id,0),(255,255,0),4)
                #     # masked = cv.line(output.warped,(p.id,height-1),(p.id,0),color,4)
                xpos = int(p.id)
                if xpos > width:
                    xpos = width-4
                elif xpos <4:
                    xpos = 4
                masked = cv2.line(masked,(xpos,height-1),(xpos,0),(255,255,0),4)
                detect = cv2.line(detect,(xpos,height-1),(xpos,0),(255,255,0),4)
            # for all the lanes - updting the 'id'
            
            index='lane_'+str(ikey)
            self.LanesParams.Lane_Detect_val[index]=p.id
            # for the practical offset to control the robot manoveoveor
            self.LanesParams.Ctrl_Ofset[index]=pre_id-p.id
            # output in mm in evrage of all lanes
            # self.LanesParams.Ctrl_Ofset_mtric[index]=self.LanesParams.Ctrl_Ofset[index]*self.LanesParams.PixelSize


        self.LanesParams.Rbt_Ofset = sum(self.LanesParams.Ctrl_Ofset.values())/(self.LanesParams.Numoflanes)#*self.LanesParams.deltalane)

        # res = np.hstack((detect,masked)) #stacking images side-by-side
        # unwarped = cv2.warpPerspective(masked, m_inv, (masked.shape[1], masked.shape[0]), flags=cv2.INTER_LINEAR)  # DEBUG
        # unwarped_snip = cv2.warpPerspective(detect, m_inv, (detect.shape[1], detect.shape[0]), flags=cv2.INTER_LINEAR)  # DEBUG
        # res = np.hstack((unwarped_snip,unwarped)) #stacking images side-by-side
        # PltShowing(res, 'PF tracking...','Perspective View Tracks', False, True)

        # cv2.imwrite('/home/chfox/prototype/Weed3DJetPrj/ref_images/Fig25BirdViewTrackrhm.jpg',unwarped)
        # cv2.imwrite('/home/chfox/prototype/Weed3DJetPrj/ref_images/Fig25BirdViewTracksniprhm.jpg',unwarped_snip)
        # self.AgthmParams.video_image = cv2.resize(res,(self.AgthmParams.frame_width,self.AgthmParams.frame_height),interpolation=cv2.INTER_AREA)
        # self.AgthmParams.out.write(self.AgthmParams.video_image)

