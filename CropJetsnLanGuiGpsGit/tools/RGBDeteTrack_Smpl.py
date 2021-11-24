from __future__ import print_function
print(__doc__)
import sys
PY3 = sys.version_info[0] == 3
import cv2
import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
# from sklearn.datasets.samples_generator import make_blobs
import matplotlib.pyplot as plt
# import pylab
import math
import statistics
# import argparse

import os, glob
# from moviepy.editor import VideoFileClip
# built-in modules
import itertools as it
from tools.LaneTrackingKF import grasstrack
from tools.CSysParams import CLanesParams,CAlgrthmParams,CCamsParams

# AgthmParams=object()

class CLneDetsTracks(CAlgrthmParams,CLanesParams):
    def __init__(self,AgthmParams,LanesParams):
        # global AgthmParams
        # AgthmParams = abb()
        self.AgthmParams = AgthmParams
        # self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.LanesParams = LanesParams        

    def center(self,points):
        """calculates centroid of a given matrix"""
        x = (points[0][0] + points[1][0] + points[2][0] + points[3][0]) / 4
        y = (points[0][1] + points[1][1] + points[2][1] + points[3][1]) / 4
        return np.array([np.float32(x), np.float32(y)], np.float32)


    def centroid_refining_new(self,centroids, bandwith):
        cen_lth = len(centroids)-1  # all list index number
        # tmp = np.median(centroids)
        inx = 0
        cent_refined = []
        if cen_lth == 1:
            cur = centroids[0]
            nxt = centroids[1]
            dis = abs(cur-nxt)
            # print('loop inx: and neighbour distance = ',inx, dis)
            if dis >= bandwith:
                cent_refined.append(cur)
                cent_refined.append(nxt)
                return cent_refined
            else:
                tmp = np.median(centroids)
                cent_refined.append(tmp)
                return cent_refined

        while (inx < cen_lth):
            if cen_lth == 1:
                cur = centroids[inx]
                cent_refined.append(cur[inx])
                inx = inx+1
            else:
                cur = centroids[inx]
                nxt = centroids[inx+1]
                dis = abs(cur-nxt)
                # print('loop inx: and neighbour distance = ',inx, dis)
                if dis > bandwith:
                    cent_refined.append(cur)
                    inx = inx+1
                    if inx == cen_lth:  # len(centroids)-1:
                        cent_refined.append(nxt)
                        break
                else:
                    tmp = (cur+nxt)*0.5
                    cent_refined.append(tmp)
                    inx = inx+2
                    # print('loop inx with close neighbour = ',inx)
                    if inx == cen_lth:
                        break

        return cent_refined
    

    def centroid_refining(self,centroids,bandwith):
        cen_lth = len(centroids)-1## all list index number
        inx=0
        cent_refined=[]
        if cen_lth==0:
                cur = centroids[0]
                # cent_refined.append(cur[inx])
                cent_refined.append([cur[0]])
                # inx=inx+1
                return cent_refined

        if cen_lth == 1:
            cur = centroids[0]
            nxt = centroids[1]
            dis=abs(cur[0]-nxt[0])
            # print('loop inx: and neighbour distance = ',inx, dis)

            if dis >= bandwith:
                cent_refined.append([cur[0]])
                cent_refined.append([nxt[0]])
                return cent_refined
            else:
                tmp = np.median(centroids)
                cent_refined.append([tmp])
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
            # print('loop inx and cen_lth  = ',inx, cen_lth) 
            # print('neighbour value = ',cur ,  nxt)
            dis=abs(cur[0]-nxt[0])
            # print('loop inx: and neighbour distance = ',inx, dis)
            if dis>bandwith:
                cent_refined.append([cur[0]]) 
                inx=inx+1 
                if  inx==cen_lth:#len(centroids)-1:          
                    cent_refined.append([nxt[0]])
                    break
            else:
                tmp=(cur[0]+nxt[0])*0.5
                cent_refined.append([tmp])
                inx=inx+2
                # print('loop inx with close neighbour = ',inx)
                if inx==cen_lth:
                    # cur = centroids[inx]
                    # cent_refined.append(cur)
                    break
        ################################
        return cent_refined
    """
    # cen_lth = len(centroids)-1  # all list index number
        # tmp = np.median(centroids)
        # inx = 0
        # cent_refined = []
        if cen_lth == 1:
            cur = centroids[0]
            nxt = centroids[1]
            dis = abs(cur-nxt)
            # print('loop inx: and neighbour distance = ',inx, dis)
            if dis >= bandwith:
                cent_refined.append([cur])
                cent_refined.append([nxt])
                return cent_refined
            else:
                tmp = np.median(centroids)
                cent_refined.append([tmp])
                return cent_refined
    ###############################

        while (inx < cen_lth):
            if cen_lth==1:
                cur = centroids[inx]
                cent_refined.append(cur[inx])
                inx=inx+1
            else:
                cur = centroids[inx]
                nxt = centroids[inx+1]
                dis=abs(cur[0]-nxt[0])
                # print('loop inx: and neighbour distance = ',inx, dis)
                if dis>bandwith:
                    cent_refined.append(cur[0]) 
                    inx=inx+1 
                    if  inx==cen_lth:#len(centroids)-1:          
                        cent_refined.append(nxt[0])
                        break
                else:
                    tmp=(cur[0]+nxt[0])*0.5
                    cent_refined.append(tmp)
                    inx=inx+2
                    # print('loop inx with close neighbour = ',inx)
                    if inx==cen_lth:
                        break
        
        return cent_refined
        """
    #line segments detection by HoughP , 
    def get_lines(self,regionofintrest,masked):    

            minLineLength = 50
            maxLineGap = 10
            lines = cv2.HoughLinesP(regionofintrest, 2, np.pi/180, 50, minLineLength, maxLineGap)
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
                    if (abs(math.degrees(orient)) > 60.0) and abs(math.degrees(orient)) < (120.0):
                        _lines_v.append(line)
                    else:
                        _lines_h.append(line)

            if _lines_h is not None:
                for line in _lines_h:
                    x1, y1, x2, y2 = line[0]
                    cv2.line(masked, (x1, y1), (x2, y2), (255, 0, 0), 5)


            ##obtain the middle points of line for the line regression later on
            _lines_mid_v = []
            lines_mid_v_arr = []
            if _lines_v is not None:
                for line in _lines_v:
                    x1, y1, x2, y2 = line[0]
                    midx=(x1+x2)*0.5
                    midy=(y1+y2)*0.5
                    _lines_mid_v.append([(midx,midy)])
                    lines_mid_v_arr.append((midx,midy))
                    cv2.line(masked, (x1, y1), (x2, y2), (0, 0, 255), 10)

            # plt.figure('Tracking...')
            # plt.title('line segments')
            # plt.imshow(masked,cmap='gray')
            # # plt.show()
            # plt.show(block=False)
            # plt.pause(0.25)
            # plt.close()


            datax=[]
            datay=[]
            if _lines_mid_v is not None:
                for point in _lines_mid_v:
                    cx=(int)(point[0][0])
                    cy=(int)(point[0][1])
                    datax.append([cx])
                    datay.append([cy])
                    cv2.circle(masked,(cx,cy),5,(0,255,255),-1) 
            
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
            # plt.scatter(datax,datay)
            # pylab.scatter(datax,datay)
            # pylab.scatter(lenx,datax)
            # pylab.savefig("/home/chfox/prototype/Weed3DJetPrj/output/lincenters.png")
            # plt.show(block=False)
            # pylab.show(block=False)
            # plt.pause(0.25)
            # plt.close()
            # pylab.close()

            # plt.show(block=False)
            # plt.pause(1)
            # plt.close()
            #clustering
            return datax, datay,_lines_mid_v



    #lanes detection by meanshift/xmean clustering
    def get_lanes(self,datax,datay,_lines_mid_v,masked,cen_hight):   
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
        print('bandwith_ested=',bandwith_ested)
        # self.AgthmParams.bandwith=75
        ms = MeanShift(bandwidth=self.LanesParams.SIMPbandwith, bin_seeding=True)
        ms.fit(datax)
        labels = ms.labels_
        centroids = ms.cluster_centers_
        # print('centroids',centroids)

        labels_unique = np.unique(labels)
        n_clusters_ = len(labels_unique)
        # print("number of estimated clusters : %d" %n_clusters_)

        cen_hight=(int)(np.median(datay))
        centroidslist=[]
        if centroids is not None:
            for cent in centroids:
                centroidslist.append([cent[0]])
                cx=(int)(cent[0])
                cy=cen_hight
                cv2.circle(masked,(cx,cy),15,(200,255,200),-1) 
        #sorting and reomve dupliation or close neighbours in 25 pixels
        # centroids=np.array(centroids)     

        centroidslist.sort()
        centroids=centroidslist
        # centroids=cv2.sort(centroids,key = lambda x: float(x[0]), reverse = True) 

        # print('centroids sorted = ',centroids)
        cent_refined = centroids.copy()
            
        while True:
            lenofrefined_prv=len(cent_refined) 
            cent_refined = self.centroid_refining(cent_refined,self.LanesParams.SIMPbandwith)
            if lenofrefined_prv==len(cent_refined):
                break

        # print('centroids refined = ',cent_refined)
        # print('lenth of centroids refined= ',len(cent_refined))
        cent_refined = np.concatenate(cent_refined)
        if cent_refined is not None:
            for cent in cent_refined:
                print('cent value = ',cent)
                cx=(int)(cent)
                cy=cen_hight
                cv2.circle(masked,(cx,cy),10,(200,0,200),-1) 

        return  cent_refined,masked,self.LanesParams.SIMPbandwith

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
                    masked = cv2.line(masked,(x,rows-1),(x,0),(200,0,200),8)
                # lefty =(int)(a+b*(cols-1))     #y=a+bx
                # masked = cv2.line(masked,(cols-1,lefty),(cx,cy),(255,0,255),8)
                else:
                    # masked = cv2.line(masked,(cols-1,lefty),(0,righty),(200,255,200),8)
                    # masked = cv2.line(masked,(cols-1,righty),(0,lefty),(200,255,200),8)
                    masked = cv2.line(masked,(lefty,rows-1),(righty,0),(200,255,200),8)
            #draw_pathlines(masked, lane_dic, color=[255, 0, 0], thickness=8, make_copy=True)
        return masked    
            # for line in get_lines(lines):
            #     leftx, boty, rightx, topy = line
            #     cv2.line(masked, (leftx, boty), (rightx,topy), (0,0,255), 5) 
            # cv2.imshow("frame", masked)
            # cv2.waitKey(100)
        # plt.figure('Tracking...')
        # plt.title('grass lane')
        # plt.imshow(masked,cmap='gray')
        # # plt.show(block=True)
        # # plt.show()
        # plt.show(block=False)
        #     # pylab.show(block=False)
        # plt.pause(0.25)
        # plt.close()
            # pylab.close()
    def perspective_transform(self,img,offset):
        
        """
        Execute perspective transform
        """
        img_size = (img.shape[1], img.shape[0])
        mask0 = np.zeros((img.shape[0], img.shape[1]), dtype="uint8")
        width = int(img.shape[1] )
        height = int(img.shape[0])
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
        
    def CameraStreamfromfile(self):
        #/home/chfox/ARWAC/Essa-0/images    
        # video_file = '/home/chfox/ARWAC/KFPFTrack/dataset/soccer_02.mp4'
        # video_file = '/home/chfox/ARWAC/KFPFTrack/dataset/crowds_zara01.avi'
        # video_file = "/home/chfox/ARWAC/Essa-0/20190206_143625.mp4"

        # args.video_file ="road_car_view.mp4"
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
                    self.AgthmParams.video_image = np.zeros((self.AgthmParams.frame_height,self.AgthmParams.frame_width,3), np.uint8)
                    self.AgthmParams.video_counter = self.AgthmParams.totalNum_frame*2
                    
        except:
            # if not then just use OpenCV default
            print("INFO: camera_stream class not found - camera input may be buffered")
            self.AgthmParams.camera = cv2.VideoCapture(0)

    def MainDeteTrack(self,orig_frame):
        
        print('Initial NumofRows = {0}'.format(self.LanesParams.SIMPNumoflaneInit)) 
        ########################################################################       
        
        self.AgthmParams.frame_num=self.AgthmParams.frame_num+1
        if self.AgthmParams.frame_num == self.AgthmParams.totalNum_frame-1:
            # recording_counter=recording_counter+1
            self.AgthmParams.camera = cv2.VideoCapture(self.AgthmParams.video_file)


        grabbed, orig_frame = self.AgthmParams.camera.read()
        if (self.AgthmParams.frame_num>self.AgthmParams.video_counter or grabbed is False):
            print ("failed to grab frame.")
            self.AgthmParams.out.release()
            cv2.destroyAllWindows()
            return
        
        ########################################################################


        print( "\n -------------------- FRAME %d --------------------" % self.AgthmParams.frame_num)    
        # width = int(orig_frame.shape[1] )
        # height = int(orig_frame.shape[0])
        # dim = (width, height)
        # print('original dimension: ', orig_frame.shape)
        # snip section of video frame of interest & show on screen

        # snip = orig_frame[20:1900,50:1050]

        row0 =self.LanesParams.SIMPRGBRow0 #= 20
        h0=self.LanesParams.SIMPRGBH0#=1880
        col0 =self.LanesParams.SIMPRGBCol0# = 50
        w0=self.LanesParams.SIMPRGBW0#=1000

        snip = orig_frame[row0:row0+h0,col0:col0+w0]

        snpwidth = int(snip.shape[1] )

        lanebandwith=self.LanesParams.SIMPbandwith    #75 #estimation or setting the initial lane wide
        Numoflanes = self.LanesParams.SIMPNumoflaneInit#.FullDTHNumRowsInit  #6
        laneoffset= self.LanesParams.SIMPOffset  #75
        LaneOriPoint = self.LanesParams.SIMPLaneOriPoint   # 5 # where the row started
        
        lstartpt = LaneOriPoint + int (lanebandwith*0.5)
        lanegap=int((snpwidth-Numoflanes*lanebandwith-LaneOriPoint*2)/(Numoflanes-1))
        self.LanesParams.deltalane= lanebandwith+lanegap

        # cv2.imshow("Snip",snip)
        # cv2.waitKey(100)
        # width = int(snip.shape[1] )
        # height = int(snip.shape[0])
        # print('snap dimensiong: ', snip.shape)
        # plt.figure('Tracking...')
        # plt.title('snap')
        # plt.imshow(snip,cmap='gray')
        # # plt.show(block=True)
        # # plt.show()
        # plt.show(block=False)
        # plt.pause(0.25)
        # plt.close()  
        # self.AgthmParams.offset=75           
        img_size = (snip.shape[1], snip.shape[0])
        m, m_inv = self.perspective_transform(snip,self.LanesParams.SIMPOffset)

        warped = cv2.warpPerspective(snip, m, img_size, flags=cv2.INTER_AREA)        
        snip=warped
        # plt.figure('warping...')
        # plt.title('warped')
        # plt.imshow(warped, cmap='gray', vmin=0, vmax=1)
        # plt.show(block=True)
        # plt.show()
        # plt.show(block=False)
        # plt.pause(0.25)
        # plt.close()         
    

        """
        https://jeffwen.com/2017/02/23/lane_finding

        """
        # mask0 = np.zeros((snip.shape[0], snip.shape[1]), dtype="uint8")
        ############################################################# 
        width = int(warped.shape[1] )
        height = int(warped.shape[0])
        """
        pts = np.array([[0, height], [self.LanesParams.SIMPOffset, 0], [width-self.LanesParams.SIMPOffset, 0], [width, height]], dtype=np.int32)
        #full image
        # pts = np.array([[0, height], [0, 0], [width, 0], [width,height]], dtype=np.int32)
        # pts = np.array([[25, 190], [275, 50], [380, 50], [575, 190]], dtype=np.int32)

        cv2.fillConvexPoly(mask0, pts, 255)
        # cv2.imshow("Mask", mask)
        # cv2.waitKey(100)
        plt.figure('Tracking...')
        plt.title('mask0')
        plt.imshow(mask0,cmap='gray')
        # plt.show(block=True)
        # plt.show()
        plt.show(block=False)
        plt.pause(0.25)
        plt.close() 
        # plt.show(block=True)
        
        # cv2.imshow("Region of Interest", masked)
        # cv2.waitKey(100)    
        plt.figure('Tracking...')
        """
        ##############################################################
        # create polygon (trapezoid) mask to select region of interest
        mask = np.zeros((snip.shape[0], snip.shape[1]), dtype="uint8")
        
        # ROI = cv2.bitwise_and(snip, snip, mask=mask)
        # # pts = np.array([[0, 1000], [25, 25], [950, 25], [1000, 1000]], dtype=np.int32)
        # #full image
        # # pts = np.array([[0, height], [0, 0], [width, 0], [width,height]], dtype=np.int32)
        pts = np.array([[0, height], [self.LanesParams.SIMPOffset, 0], [width-self.LanesParams.SIMPOffset, 0], [width, height]], dtype=np.int32)
        cv2.fillConvexPoly(mask, pts, 255)
        # # cv2.imshow("Mask", mask)
        # # cv2.waitKey(100)
        # plt.figure('Tracking...')
        # plt.title('mask')
        # plt.imshow(mask,cmap='gray')
        # # plt.show(block=True)
        # # plt.show()
        # plt.show(block=False)
        # plt.pause(0.25)
        # plt.close()
        cen_hight=(int)(0.382*height)
        # print('masked dimensiong: ', mask.shape)
        


        # apply mask and show masked image on screen
        masked = np.copy(warped)#cv2.bitwise_and(snip, snip, mask=mask)

    
        # plt.figure('Tracking...')
        # plt.title('hsv_frame')
        hsv_frame = cv2.cvtColor(snip, cv2.COLOR_BGR2HSV)
        # plt.imshow(hsv_frame,cmap='gray')
        # # plt.show(block=True)
        # # plt.show()
        # plt.show(block=False)
        # plt.pause(0.25)
        # plt.close() 
        gclor_l= self.LanesParams.HSVLowGreenClor_Smpl 
        gclor_h= self.LanesParams.HSVHighGreenClor_Smpl

        gSatur_l= self.LanesParams.HSVLowGreenSatur_Smpl 
        gSatur_h= self.LanesParams.HSVHighGreenSatur_Smpl

        gval_l= self.LanesParams.HSVLowGreenVal_Smpl 
        gval_h= self.LanesParams.HSVHighGreenVal_Smpl

        # low_green = np.array([42, 25, 25])
        # high_green = np.array([106, 225, 225])
        low_green = np.array([gclor_l, gSatur_l, gval_l])
        high_green = np.array([gclor_h, gSatur_h, gval_h])


        # Green color
        # low_green = np.array([50, 60, 60])
        # high_green = np.array([80, 255, 255])

        # low_green = np.array([25, 52, 72])
        # high_green = np.array([102, 255, 255])

        # low_yellow = np.array([18, 94, 140])
        # up_yellow = np.array([48, 255, 255])

        # mask = cv2.inRange(hsv_frame, low_green, up_green)

        hsv_green_mask = cv2.inRange(hsv_frame, low_green, high_green)

        # plt.figure('Tracking...')
        # plt.title('hsv_green_mask')
        # plt.imshow(hsv_green_mask,cmap='gray')
        # # plt.show(block=True)
        # # plt.show()
        # plt.show(block=False)
        # plt.pause(0.25)
        # plt.close() 


        #blured image to help with edge detection
        # blurred = cv2.GaussianBlur(hsv_green_mask,(21,21),0); 
        """
        src – Source 8-bit or floating-point, 1-channel or 3-channel image.
        dst – Destination image of the same size and type as src .
        d – Diameter of each pixel neighborhood that is used during filtering. If it is non-positive, it is computed from sigmaSpace .
        sigmaColor – Filter sigma in the color space. A larger value of the parameter means that farther colors within the pixel neighborhood (see sigmaSpace ) will be mixed together, resulting in larger areas of semi-equal color.
        sigmaSpace – Filter sigma in the coordinate space. A larger value of the parameter means that farther pixels will influence each other as long as their colors are close enough (see sigmaColor ). When d>0 , it specifies the neighborhood size regardless of sigmaSpace . Otherwise, d is proportional to sigmaSpace .

        """
        blurred = cv2.bilateralFilter(hsv_green_mask, 5, 75, 75)
        # plt.figure('Tracking...')
        # plt.title('blurred')
        # plt.imshow(blurred,cmap='gray')
        # # plt.show(block=True)
        # # plt.show()
        # plt.show(block=False)
        # plt.pause(0.25)
        # plt.close() 


        # cv2.imshow("blurred", blurred)
        # cv2.waitKey(100)
        # cv2.imwrite('./output/blurred.png',blurred) 

        # morphological operation kernal - isolate the row - the only purpose
        # small kernel has thining segmentation , which may help with later hough transform, then group the segments of line
        #https://github.com/opencv/opencv/blob/master/samples/python/tutorial_code/imgProc/morph_lines_detection/morph_lines_detection.py
        # 1st erode then dialted , in most case like this :   erode - dilate in above link
        rows=3
        cols=3
        kernel = np.ones((rows,cols),dtype=np.uint8)
        erosion = cv2.erode(blurred,kernel,iterations = 1) #thinning, the lines
        # plt.figure('Tracking...')
        # plt.title('erosion')
        # plt.imshow(erosion,cmap='gray')
        # plt.show(block=False)
        # plt.pause(0.25)
        # plt.close()

        #######################################################
        # Opening = erosion followed by dilation
        opening = cv2.morphologyEx(blurred, cv2.MORPH_OPEN, kernel)
        # plt.figure('Tracking...')
        # plt.title('opening')
        # plt.imshow(opening,cmap='gray')
        # plt.show(block=False)
        # plt.pause(0.25)
        # plt.close()
        # plt.show()
        # plt.show(block=False)
        # plt.pause(1)
        # plt.close()

        # cv2.imshow("opening", opening)
        # cv2.waitKey(100)
        # cv2.imwrite('./output/green.png',opening)
        
        # Closing  = Dilation followed by Erosion
        closing = cv2.morphologyEx(blurred, cv2.MORPH_CLOSE, kernel)
        # plt.figure('Tracking...')
        # plt.title('closing')
        # plt.imshow(blurred,cmap='gray')   
        # plt.show(block=False)
        # plt.pause(1)
        # plt.close()
        # plt.show(block=False)
        # plt.pause(0.25)
        # plt.close()
        # plt.show()

        blurred=closing #erosion#
        green = cv2.bitwise_and(warped, warped, mask=blurred)
        # plt.figure('Tracking...')
        # plt.title('lanes area')
        # plt.imshow(green,cmap='gray')
        # plt.show(block=False)
        # plt.pause(0.25)
        # plt.close()

        # cv2.imwrite('./output/green.png',green)

        
        edges = cv2.Canny(blurred, 50, 200) #150
        # plt.figure('Tracking...')
        # plt.title('edges')
        # plt.imshow(edges,cmap='gray')
        # plt.show(block=False)
        # plt.pause(0.25)
        # plt.close()
        # cv2.imwrite('./output/edges.png',edges)
        ####################################################
        # apply mask and on canny image on screen
        regionofintrest = cv2.bitwise_and(edges, edges, mask=mask) # mask, mask0
        #get line segments from region of interest
        datax,datay,_lines_mid_v=self.get_lines(regionofintrest,masked)
        #get lane sortd by finding the center of each lane
        cent_refined,masked,bandwith=self.get_lanes(datax,datay,_lines_mid_v,masked,cen_hight)      
        #get dictionary centroid <=> lane : (x,y)
        lane_dic= self.get_dic(cent_refined,_lines_mid_v,masked,bandwith)
    
        masked=self.draw_display(masked,lane_dic,cen_hight,width)
        if self.AgthmParams.ToIntializeSetting is True:
            rows = int(masked.shape[0])
            for la in range(lstartpt,snpwidth,self.LanesParams.deltalane):                 
                self.AgthmParams.video_image = cv2.line(masked,(la,rows-1),(la,0),(0,0,0),10)
            return   

        # for pt in cent_refined:
        if self.AgthmParams.firstFrame is True:
            for key, coords in lane_dic.items(): # try to get each column
                lth=len(coords)
                lane_centers=[]
                id=int(key)
                if (lth>0):            
                    for pt in coords:      
                        lane_centers.append([pt])
                        # print('pt = : ', pt)         
                
                print('lane_centers = : ', lane_centers)      
                cx=(int)(key)
                cxy= np.median(lane_centers,axis=0)#
                track_pt=np.array((cxy[0][0],cxy[0][1]),np.float32)
                print('track_pt = : ', track_pt) 
                x,y,w,h=(int)(cxy[0][0]),(int)(cxy[0][1]),20,20


                if self.AgthmParams.firstFrame is True:
                    # for id in range(5,1000,195):  #5,200,195,580, 775,970 
                    #     track_pt=[id,cen_hight]
                    #     _lines_mid_v=[(id, cen_hight)]
                    #     self.AgthmParams.grasslanes[id]=grasstrack(id,masked,track_pt,(x,y,w,h),_lines_mid_v,bandwith)
                    # deltalane= lanebandwith+lanegap
                    # print('Lane Gap = {0}, Lane Width = {1}, Total Lane width = {2}, loop Deta = {3}'.format(lanegap,self.AgthmParams.FullRGBbandwith,snpwidth,deltalane))
                    print('Lane Gap = {0}, Lane Width = {1}, Total Lane width = {2}, loop Deta = {3}'.format(lanegap,lanebandwith,snpwidth,self.LanesParams.deltalane))

                    NumofSettingID=0
                    # for la in range(self.LanesParams.FullRGBLaneOriPoint,snpwidth,deltalane):  

                    for la in range(lstartpt,snpwidth,self.LanesParams.deltalane):                   
                        track_pt=[la,cen_hight]
                        _lines_mid_v=[(la, cen_hight)]
                        self.AgthmParams.grasslanes[la]=grasstrack(la,masked,track_pt,(x,y,w,h),_lines_mid_v,bandwith)
                        NumofSettingID=NumofSettingID+1
                    print('total number of lanes tracked = {0}'.format(NumofSettingID))

            self.AgthmParams.firstFrame=False
            # frame_num=frame_num+1



        if self.AgthmParams.firstFrame is False:
            print('\n kalman tracking points ...........')
            for i, p in self.AgthmParams.grasslanes.items():#.iteritems(): 
                found_flg=False # tracking the updated lanes  
                for key, coords in lane_dic.items(): # try to get each column
                    lth=len(coords)
                    lane_centers=[]
                    id=int(key)
                    if (lth>0):            
                        for pt in coords:      
                            lane_centers.append([pt])
                            # print('pt = : ', pt)         
                    # print('lane_centers = : ', lane_centers)      
                    cx=(int)(key)
                    cxy= np.median(lane_centers,axis=0)#
                    track_pt=np.array((cxy[0][0],cxy[0][1]),np.float32)
                    print('track_pt = : ', track_pt) 
                    x,y,w,h=(int)(cxy[0][0]),(int)(cxy[0][1]),20,20
                    # (self.LanesParams.deltalane*0.618):                       #bandwith
                    if abs(p.id-id) <= bandwith:
                        masked = p.update(masked, track_pt, (x, y, w, h))
                        p.id = int(0.5*(p.id+id))  # int(p.center[0])

                    
                        #unwarped = cv2.warpPerspective(masked, m_inv, (masked.shape[1], masked.shape[0]), flags=cv2.INTER_LINEAR)  # DEBUG
                        # plt.figure('kalman tracking...')
                        # plt.title('track being updated')
                        # plt.imshow(masked, cmap='gray', vmin=0, vmax=1)
                        # plt.show(block=False)
                        # plt.pause(0.25)
                        # plt.close()
                        found_flg=True
                        break
                if found_flg==False:
                    print("Lande id: %d is not updated!" % p.id)
                    width = int(masked.shape[1])
                    height = int(masked.shape[0])
                    pred_posx=int(p.prediction[0])
                    # print("pred_posx={0:d}, pred_posy={1:d}".format(p.prediction[0],p.prediction[1]))
                    # print("pred_posx= %d, pred_posy= %d" %(p.prediction[0],p.prediction[1]))
                    if abs(p.id-pred_posx)<(int)(self.LanesParams.deltalane*0.191):            # bandwith
                        masked = cv2.line(masked,(pred_posx,height-1),(pred_posx,0),(255,255,0),4)
                        p.id=pred_posx
                    else:
                        masked = cv2.line(masked,(p.id,height-1),(p.id,0),(255,255,0),4)                    
                    # plt.figure('kalman tracking...')
                    # plt.title('track no updated')
                    # plt.imshow(masked,cmap='gray')
                    # plt.show(block=False)
                    # plt.pause(0.25)
                    # plt.close()

            unwarped = cv2.warpPerspective(masked, m_inv, (masked.shape[1], masked.shape[0]), flags=cv2.INTER_LINEAR)  # DEBUG
            # plt.figure('kalman tracking...')
            # plt.title('Lane Following')
            # cv2.resize(oriimg,None,fx=0.5,fy=0.5,interpolation=cv2.INTER_CUBIC)
            # img = cv2.resize(oriimg,(200,300),interpolation=cv2.INTER_AREA)
            self.AgthmParams.video_image = cv2.resize(unwarped,(self.AgthmParams.frame_width,self.AgthmParams.frame_height),interpolation=cv2.INTER_AREA)
            self.AgthmParams.out.write(self.AgthmParams.video_image)
            # plt.imshow(unwarped, cmap='gray', vmin=0, vmax=1)
            # plt.show(block=False) 
            # plt.pause(0.25)  
            # self.AgthmParams.out.write(self.AgthmParams.video_image)                      
        ################################               
        ################################     

        key = cv2.waitKey(1)
        plt.close('all') 
        
        if key == 27:
            return
    
        self.AgthmParams.out.release()
        cv2.destroyAllWindows()

    # if __name__ == "__main__":
    # main()
