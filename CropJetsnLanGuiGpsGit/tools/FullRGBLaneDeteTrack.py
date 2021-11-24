from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from pickle import TRUE
print(__doc__)
import sys
PY3 = sys.version_info[0] == 3
import cv2
import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
# from sklearn.datasets.samples_generator import make_blobs
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
# import pylab
import math
import statistics
# import argparse
from collections import OrderedDict
import os, glob
# from moviepy.editor import VideoFileClip
# built-in modules
import itertools as it
from tools.LaneTrackingKF import grasstrack
from tools.CSysParams import CLanesParams,CAlgrthmParams,CCamsParams
from tools.HSVRGBSegt4HT import PltShowing,cvSaveImages,HSVFullEdgeProc, RGBColorSegmtFull#, 

# from tools.videofig import videofig
# NUM_IMAGES = 100
# PLAY_FPS = 100  # set a large FPS (e.g. 100) to test the fastest speed our script can achieve
# SAVE_PLOTS = False  # whether to save the plots in a directory


# AgthmParams=object()

class CFullRGBLneDetsTracks(CLanesParams):
    def __init__(self,LanesParams):
        # global AgthmParams
        # AgthmParams = abb()
        # self.AgthmParams = AgthmParams
        # self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.LanesParams = LanesParams        

    # def center(self,points):
    #     """calculates centroid of a given matrix"""
    #     x = (points[0][0] + points[1][0] + points[2][0] + points[3][0]) / 4
    #     y = (points[0][1] + points[1][1] + points[2][1] + points[3][1]) / 4
    #     return np.array([np.float32(x), np.float32(y)], np.float32)
    

    def centroid_refining_new(self,centroids,bandwith):
        cen_lth = len(centroids)-1  # all list index number
        inx = 0
        cent_refined = []

        if cen_lth==0:
            cur = centroids[0]
            # cent_refined.append(cur[inx])
            cent_refined.append(cur)
            # inx=inx+1
            return cent_refined

        if cen_lth == 1:
            cur = centroids[0]
            nxt = centroids[1]
            # dis=abs(cur[0]-nxt[0])
            dis=abs(cur-nxt)
            # print('loop inx: and neighbour distance = ',inx, dis)
            if dis >= bandwith:
                cent_refined.append(cur)
                cent_refined.append(nxt)
                return cent_refined
            else:
                tmp = (cur + nxt)*0.5# np.median(centroids)
                cent_refined.append(tmp)
                return cent_refined

        while (inx< cen_lth):
            # if cen_lth==0:
            #     cur = centroids[0]
            #     # cent_refined.append(cur[inx])
            #     cent_refined.append(cur[0])
            #     # inx=inx+1
            #     break
            # else:
            cur = centroids[inx]
            nxt = centroids[inx+1]
            dis=abs(cur-nxt)
            # print('loop inx: and neighbour distance = ',inx, dis)
            if dis>bandwith:
                cent_refined.append(cur) 
                inx=inx+1 
                if  inx==cen_lth:#len(centroids)-1:          
                    cent_refined.append(nxt)
                    break
            else:
                tmp=(cur+nxt)*0.5
                cent_refined.append(tmp)
                inx=inx+2
                # print('loop inx with close neighbour = ',inx)
                if inx==cen_lth:
                    break

        return cent_refined
    #line segments detection by HoughP , 
    def get_lines(self,regionofintrest,masked):    

        minLineLength = self.LanesParams.minLineLength    #30
        maxLineGap = self.LanesParams.maxLineGap#4
        max_slider = 50
        lines = cv2.HoughLinesP(regionofintrest, 2, np.pi/180, max_slider, minLineLength, maxLineGap)
        # lines = cv2.HoughLinesP(edges, 2, np.pi/180, 100)

        # sort
        _lines_h = [] #horizontal line
        _lines_v = [] #vertical line
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                orient = math.atan2((y1-y2),(x1-x2))
                # orient = math.atan2((line[0][1]-line[1][1]),(line[0][0]-line[1][0]))
                # orient = math.atan2((line[0]-line[2]),(line[1]-line[3]))
                if (abs(math.degrees(orient)) > 75.0) and abs(math.degrees(orient)) < (105.0):
                    _lines_v.append(line)
                else:
                    _lines_h.append(line)

        if _lines_h is not None:
            for line in _lines_h:
                x1, y1, x2, y2 = line[0]
                cv2.line(masked, (x1, y1), (x2, y2), (255, 0, 0), 3)


        ##obtain the middle points of line for the line regression later on
        _lines_mid_v = []
        if _lines_v is not None:
            for line in _lines_v:
                x1, y1, x2, y2 = line[0]
                midx=(x1+x2)*0.5
                midy=(y1+y2)*0.5
                _lines_mid_v.append([(midx,midy)])
                cv2.line(masked, (x1, y1), (x2, y2), (0, 0, 255), 3)


        PltShowing(masked,'Tracking...','line segments',False,False)

        datax=[]
        datay=[]
        if _lines_mid_v is not None:
            for point in _lines_mid_v:
                cx=(int)(point[0][0])
                cy=(int)(point[0][1])
                datax.append([cx])
                datay.append([cy])
                cv2.circle(masked,(cx,cy),2,(0,255,255),-1) 

            #clustering
        return datax, datay,_lines_mid_v



    #lanes detection by meanshift/xmean clustering
    def get_lanes(self,datax,datay,_lines_mid_v,masked,cen_hight,bandwith): 
        
        if len(datax)<2:
            cent_refined = []
            return  cent_refined,masked,bandwith  
        """ xmean :   
        #The initial value is determined by k-means ++.
        init_center = pyclustering.cluster.xmeans.kmeans_plusplus_initializer(datax, 4).initialize()
        xm = pyclustering.cluster.xmeans.xmeans(datax, init_center, ccore=True)
        xm.process() # cluster

        # When ccore = True, c / c ++ is used for processing, and it seems to be faster.
        # For visualization, it is easy to use the functions provided by pyclustering.
        print('one loop ##################/n')
        clusters = xm.get_clusters()
        centroids= xm.get_centers()
        print('centroids = ',centroids)
        print('lenth of centroids = ',len(centroids))
        """   
        # meanshift:
        # print('datax',datax)
        #The following bandwidth can be automatically detected using    # 
        bandwith_ested = estimate_bandwidth(datax, quantile=0.2, n_samples=20)
        # print('bandwith_ested=',bandwith_ested)
        
        ms = MeanShift(bandwidth=bandwith, bin_seeding=True)
        ms.fit(datax)
        labels = ms.labels_
        centroids = ms.cluster_centers_
        # print('lanebandwith = {0}, centroids= {1}'.format(bandwith,centroids))

        labels_unique = np.unique(labels)
        n_clusters_ = len(labels_unique)
        # print("number of estimated clusters : %d" %n_clusters_)

        cen_hight=(int)(np.median(datay))
        centroidslist=[]
        if centroids is not None:
            for cent in centroids:
                centroidslist.append(cent[0])
                cx=(int)(cent[0])
                cy=cen_hight
                cv2.circle(masked,(cx,cy),10,(200,255,200),-1) 
        #sorting and reomve dupliation or close neighbours in 25 pixels
        # centroids=np.array(centroids)     

        centroidslist.sort()
        centroids=centroidslist
        # centroids=cv2.sort(centroids,key = lambda x: float(x[0]), reverse = True) 

        # print('centroids sorted = ',centroids)
        cent_refined = centroids.copy()
        # lenofOriCentrs = len(cent_refined)
        # if lenofOriCentrs > 1:
        while True:
            lenofrefined_prv=len(cent_refined) 
            cent_refined = self.centroid_refining_new(cent_refined,bandwith)
            if lenofrefined_prv==len(cent_refined):
                break

        # print('centroids refined = ',cent_refined)
        # print('lenth of centroids refined= ',len(cent_refined))
        if cent_refined is not None:
            for cent in cent_refined:
                # print('cent value = ',cent)
                # if lenofOriCentrs==1:
                    # cent_refined=[]
                    # cent_refined.append(cent[0])
                    # cx=(int)(cent[0])              
                # else:
                cx=(int)(cent)
                    
                cy=cen_hight
                cv2.circle(masked,(cx,cy),8,(200,0,200),-1) 

        return  cent_refined,masked,bandwith
            # return [l[0] for l in lines_in]

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


    def draw_display(self,masked,lane_dic,cen_hight,width):
        
        cols = int(masked.shape[1] )
        rows = int(masked.shape[0])
        for key, coords in lane_dic.items(): # try to get each column
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
                    # print('denominator = : ',denominator)  

                a= (sumy*sumx2-sumx*sumxy)/denominator         
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
                    masked = cv2.line(masked,(x,rows-1),(x,0),(200,0,200),8)
                # lefty =(int)(a+b*(cols-1))     #y=a+bx
                # masked = cv2.line(masked,(cols-1,lefty),(cx,cy),(255,0,255),8)
                else:
                    # masked = cv2.line(masked,(cols-1,lefty),(0,righty),(200,255,200),8)
                    # masked = cv2.line(masked,(cols-1,righty),(0,lefty),(200,255,200),8)
                    masked = cv2.line(masked,(lefty,rows-1),(righty,0),(200,255,200),8)
            #draw_pathlines(masked, lane_dic, color=[255, 0, 0], thickness=8, make_copy=True)
        PltShowing(masked,'draw-display...','detected lane',True,True)
        return masked

    def perspective_transform(self,img_size,offset):
        
        """
        Execute perspective transform
        """
        # img_size = (img.shape[1], img.shape[0])
        # mask0 = np.zeros((img.shape[0], img.shape[1]), dtype="uint8")
        width = img_size[0]#int(img.shape[1] )
        height = img_size[1]# int(img.shape[0])
        dim = (width, height)
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
        
    """    
    def CameraStreamfromfile(self):
        #/home/chfox/ARWAC/Essa-0/images    
        # video_file = '/home/chfox/ARWAC/KFPFTrack/dataset/soccer_02.mp4'
        # print('ocv: ={0}'.format(self.AgthmParams.video_ocv))
        try:
            # to use a non-buffered camera stream (via a separate thread)
            if not(self.LanesParams.video_ocv):
                # import camera_stream
                # self.LanesParams.camera = camera_stream.CameraVideoStream()
                print('NO Video files!')
            else:
                # Check if camera opened successfully
                self.LanesParams.camera = cv2.VideoCapture(self.LanesParams.video_ocv) 
                if (self.LanesParams.camera.isOpened()== False): 
                    print("Error opening video stream or file")
                    # cap = cv2.VideoCapture() # not needed for video files
                else:
                    self.LanesParams.totalNum_frame =int(self.LanesParams.camera.get(cv2.CAP_PROP_FRAME_COUNT))
                    print('frame total numer={0}, width={1}, Height={2}'.format(self.LanesParams.totalNum_frame,self.LanesParams.frame_width,self.LanesParams.frame_height))
                    self.LanesParams.video_image = np.zerosk((self.LanesParams.frame_height,self.LanesParams.frame_width,3), np.uint8)
                    self.LanesParams.video_counter = self.LanesParams.totalNum_frame*2
                    
        except:
            # if not then just use OpenCV default
            print("INFO: camera_stream class not found - camera input may be buffered")
            # self.AgthmParams.camera = cv2.VideoCapture(0)
    """

    def MainDeteTrack(self,orig_frame):
        
        # print('Initial NumofRows Setting = {0}'.format(self.LanesParams.Numoflanes)) 
        ########################################################################
        # self.LanesParams.frame_num=self.LanesParams.frame_num+1
        """
        if self.AgthmParams.isavi ==True:     
        
            if self.AgthmParams.frame_num == self.AgthmParams.totalNum_frame-1:
                # recording_counter=recording_counter+1
                self.AgthmParams.camera = cv2.VideoCapture(self.AgthmParams.video_ocv)
            grabbed, orig_frame = self.AgthmParams.camera.read()
            while (self.AgthmParams.frame_num<200):
                self.AgthmParams.frame_num+=1
                grabbed, orig_frame = self.AgthmParams.camera.read()            
                continue        

            if (self.AgthmParams.frame_num>self.AgthmParams.video_counter or grabbed is False):
                print ("failed to grab frame.")
                self.AgthmParams.out.release()
                cv2.destroyAllWindows()

        """
        self.LanesParams.filter_start=self.LanesParams.filter_start+1
        print( "\n -------------------- FRAME %d --------------------" % self.LanesParams.frame_num) 

        row0 =self.LanesParams.row0 #= 0
        h0=self.LanesParams.h0#=375
        col0 =self.LanesParams.col0# = 200
        w0=self.LanesParams.w0#=300
        PltShowing(orig_frame,'Camera Output...','Origin', False,False)

        snip = orig_frame[row0:row0+h0,col0:col0+w0]
        PltShowing(snip,'Camera Output...','snip', False,False)

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

        # m, m_inv = self.perspective_transform(snip,laneoffset)

        warped = cv2.warpPerspective(snip, m, img_size, flags=cv2.INTER_LINEAR) 
        PltShowing(warped,'warping...','warped', False, False)
        cen_hight=(int)(0.382*warped.shape[0])
        # print('warped dimension: ', warped.shape)       
        #################################                    
 
        width = int(warped.shape[1] )
        height = int(warped.shape[0])
        # create polygon (trapezoid) mask to select region of interest
        # mask = np.zeros((snip.shape[0], snip.shape[1]), dtype="uint8")
        # pts = np.array([[0, height], [0, 0], [width, 0], [width,height]], dtype=np.int32)
        # # pts = np.array([[25, 190], [275, 50], [380, 50], [575, 190]], dtype=np.int32)
        # cv2.fillConvexPoly(mask, pts, 255)
  
        masked = np.copy(warped)#cv2.bitwise_and(snip, snip, mask=mask)
        detect = np.copy(warped) 
        # hsv_frame = cv2.cvtColor(snip, cv2.COLOR_BGR2HSV)
        """
        gclor_l= self.LanesParams.HSVLowGreenClor 
        gclor_h= self.LanesParams.HSVHighGreenClor

        gSatur_l= self.LanesParams.HSVLowGreenSatur 
        gSatur_h= self.LanesParams.HSVHighGreenSatur

        gval_l= self.LanesParams.HSVLowGreenVal 
        gval_h= self.LanesParams.HSVHighGreenVal

        # low_green = np.array([42, 25, 25])
        # high_green = np.array([106, 225, 225])
        low_green = np.array([gclor_l, gSatur_l, gval_l])
        high_green = np.array([gclor_h, gSatur_h, gval_h])
        """

        low_green = self.LanesParams.low_green #np.array([gclor_l, gSatur_l, gval_l])
        high_green = self.LanesParams.high_green  #np.array([gclor_h, gSatur_h, gval_h])
        regionofintrest = HSVFullEdgeProc(warped,low_green,high_green,'Tx2') 

        datax,datay,_lines_mid_v=self.get_lines(regionofintrest,masked)
       
        #get lane sortd by finding the center of each lane
        cent_refined,masked,bandwith=self.get_lanes(datax,datay,_lines_mid_v,masked,cen_hight,lanebandwith)      
        #get dictionary centroid <=> lane : (x,y)
        lane_dic= self.get_dic(cent_refined,_lines_mid_v,masked,lanebandwith)  


        ####################################################   
        #get dictionary centroid <=> lane : (x,y)
        lane_dic= self.get_dic(cent_refined,_lines_mid_v,masked,bandwith)
    
        masked=self.draw_display(masked,lane_dic,cen_hight,width)

        if self.LanesParams.ToIntializeSetting is True:
            rows = int(masked.shape[0])
            for la in range(laneStartPo,snpwidth+1,self.LanesParams.deltalane):                 
                self.LanesParams.video_image = cv2.line(masked,(la,rows-1),(la,0),(0,0,0),10)

            self.LanesParams.ToIntializeSetting = False
            return        
        ###########################################

        # for pt in cent_refined:
        if self.LanesParams.firstFrame is True:
            # deltalane= lanebandwith+lanegap
            deltalane= lanegap
            print('Lane Gap = {0}, Lane Width = {1}, Total Lane width = {2}, loop Deta = {3}'.format(lanegap,lanebandwith,snpwidth,deltalane))
            NumofSettingID=0
            # for la in range(LaneOriPoint,snpwidth,deltalane):  
            for la in range(laneStartPo,snpwidth-laneStartPo+1,self.LanesParams.deltalane):           
                track_pt=[la,cen_hight]
                _lines_mid_v=[]#[(la, cen_hight)]
                x,y,w,h=la,cen_hight,10,10
                self.LanesParams.grasslanes[la]=grasstrack(la,masked,track_pt,(x,y,w,h),_lines_mid_v,lanebandwith,deltalane)
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
            self.LanesParams.firstFrame=False

                
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
        self.LanesParams.Lane_Detect_val={}
        if self.LanesParams.firstFrame is False:
            # print('\n kalman tracking points ...........')
            for ikey, p in self.LanesParams.grasslanes.items():#.iteritems(): 
                found_flg=False # tracking the updated lanes  
                # keep last step id
                pre_id = p.id
                pre_predic_id = p.pre_updated_id

                for key, coords in lane_dic.items(): # try to get each column
                    lth=len(coords)
                    lane_centers=[]
                    # id=int(key)
                    if (lth>0):            
                        for pt in coords:      
                            lane_centers.append([pt])
                            # print('pt = : ', pt)         
                    # print('lane_centers = : ', lane_centers)      
                    # cx=(int)(key)
                    cxy= np.mean(lane_centers,axis=0)#
                    track_pt=np.array((cxy[0][0],cxy[0][1]),np.float32)
                    # print('track_pt = : ', track_pt) 
                    x,y,w,h=(int)(cxy[0][0]),(int)(cxy[0][1]),10,10
                    if abs(ikey-key)<self.LanesParams.deltalane*0.618:#(lanebandwith*1.618):                       
                       masked = p.update(masked,track_pt,(x,y,w,h))                       
                       p.id=ikey*0.382+key*0.618 # must keep here before later applying kalman filter
                       # p.id=key  # for the wide lane , trust detection
                       if self.LanesParams.filter_start>=self.LanesParams.upd_thrsh: #abs(p.id-p.center[0])<(lanebandwith*0.1545):
                           p.id = ikey*0.382 + p.center[0][0]*0.618
                           p.pre_updated_id = p.prediction[0][0]
                        #    p.id = ikey*0.5 + p.center[0]*0.5
                        #    detect = p.update(detect,track_pt,(x,y,w,h))
                       xpos = int(p.id)
                       if xpos > width:
                           xpos = width-4
                       elif xpos <4:
                           xpos = 4
                       detect = cv2.line(detect,(xpos,height-1),(xpos,0),(255,255,0),4)                       
                       lane_dic.pop(key)
                       found_flg=True
                       break
                if found_flg==False:
                    print("Lande id: %d is not updated!" % p.id)
                    if self.LanesParams.filter_start>self.LanesParams.upd_thrsh:
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
                # self.LanesParams.Ctrl_Ofset[index]=self.LanesParams.Ctrl_Ofset[index]*self.LanesParams.PixelSize
        
        self.LanesParams.Rbt_Ofset = sum(self.LanesParams.Ctrl_Ofset.values())/(self.LanesParams.Numoflanes)#*self.LanesParams.deltalane)

        
        res = np.hstack((detect,masked)) #stacking images side-by-side
        unwarped = cv2.warpPerspective(masked, m_inv, (masked.shape[1], masked.shape[0]), flags=cv2.INTER_LINEAR)  # DEBUG
        unwarped_snip = cv2.warpPerspective(detect, m_inv, (detect.shape[1], detect.shape[0]), flags=cv2.INTER_LINEAR)  # DEBUG
        res = np.hstack((unwarped_snip,unwarped)) #stacking images side-by-side
        # cv2.imwrite('/home/chfox/prototype/Weed3DJetPrj/ref_images/Fig25BirdViewTrackrhm.jpg',unwarped)
        # cv2.imwrite('/home/chfox/prototype/Weed3DJetPrj/ref_images/Fig25BirdViewTracksniprhm.jpg',unwarped_snip)
        # self.AgthmParams.video_image = cv2.resize(res,(self.AgthmParams.frame_width,self.AgthmParams.frame_height),interpolation=cv2.INTER_AREA)
        # self.AgthmParams.out.write(self.AgthmParams.video_image)
        # PltShowing(unwarped_snip, 'kalman tracking...','PerspectivePltShowing View Tracks', False, True)
        # self.AgthmParams.video_image = None
        self.LanesParams.video_image = plt.imshow(res)
        plt.pause(.1)
        plt.draw()
        
