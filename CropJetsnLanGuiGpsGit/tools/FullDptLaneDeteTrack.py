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

class CFullDpthLneDetsTracks(CAlgrthmParams,CLanesParams):
    def __init__(self,AgthmParams,LanesParams):
        # global AgthmParams
        # AgthmParams = abb()
        self.AgthmParams = AgthmParams
        # self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.LanesParams = LanesParams   
        self.font = cv2.FONT_HERSHEY_SIMPLEX
     

    def center(self,points):
        """calculates centroid of a given matrix"""
        x = (points[0][0] + points[1][0] + points[2][0] + points[3][0]) / 4
        y = (points[0][1] + points[1][1] + points[2][1] + points[3][1]) / 4
        return np.array([np.float32(x), np.float32(y)], np.float32)

    

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
        return cent_refined

    """
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
        """                    
        # return cent_refined
    #line segments detection by HoughP , 
    def get_lines(self,regionofintrest,masked):    

            minLineLength = 50
            maxLineGap = 6
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
            if _lines_v is not None:
                for line in _lines_v:
                    x1, y1, x2, y2 = line[0]
                    midx=(x1+x2)*0.5
                    midy=(y1+y2)*0.5
                    _lines_mid_v.append([(midx,midy)])
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
            # lenx = np.linspace(0, widthx, widthx)
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
        ms = MeanShift(bandwidth=self.LanesParams.FullDptbandwith, bin_seeding=True)
        ms.fit(datax)
        labels = ms.labels_
        centroids = ms.cluster_centers_
        # print('centroids',centroids)

        labels_unique = np.unique(labels)
        n_clusters_ = len(labels_unique)
        print("number of estimated clusters : %d" %n_clusters_)

        cen_hight=(int)(np.median(datay))
        centroidslist=[]
        if centroids is not None:
            for cent in centroids:
                centroidslist.append([cent[0]])
                cx=(int)(cent[0])
                cy=cen_hight
                cv2.circle(masked,(cx,cy),10,(200,255,200),-1) 
        #sorting and reomve dupliation or close neighbours in 25 pixels
        # centroids=np.array(centroids)     

        centroidslist.sort()
        centroids=centroidslist
        # centroids=cv2.sort(centroids,key = lambda x: float(x[0]), reverse = True) 

        # print('centroids sorted = ',centroids)
        cent_refined=centroids.copy()
            
        while True:
            lenofrefined_prv=len(cent_refined) 
            cent_refined = self.centroid_refining(cent_refined,self.LanesParams.FullDptbandwith)
            if lenofrefined_prv==len(cent_refined):
                break

        # print('centroids refined = ',cent_refined)
        # print('lenth of centroids refined= ',len(cent_refined))
        cent_refined = np.concatenate(cent_refined)
        if cent_refined is not None:
            for cent in cent_refined:
                # print('cent value = ',cent)
                cx=(int)(cent)
                cy=cen_hight
                cv2.circle(masked,(cx,cy),10,(200,0,200),-1) 

        return  cent_refined,masked,self.LanesParams.FullDptbandwith

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

        print('ocv: ={0}, dpt = {1}'.format(self.AgthmParams.video_ocv, self.AgthmParams.video_dpt))
        try:
            # to use a non-buffered camera stream (via a separate thread)
            if not(self.AgthmParams.video_ocv or self.AgthmParams.video_dpt):
                import camera_stream
                # self.AgthmParams.camera = camera_stream.CameraVideoStream()
                self.AgthmParams.camera_l = camera_stream.CameraVideoStream()
                self.AgthmParams.camera_d = camera_stream.CameraVideoStream()
            else:
                # Check if camera opened successfully
                self.AgthmParams.camera_l = cv2.VideoCapture(self.AgthmParams.video_ocv) 
                self.AgthmParams.camera_d = cv2.VideoCapture(self.AgthmParams.video_dpt)

                if (self.AgthmParams.camera_l.isOpened()== False or self.AgthmParams.camera_d.isOpened()== False): 
                    print("Error opening video stream or file")
                else:
                    self.AgthmParams.totalNum_frame =int(self.AgthmParams.camera_d.get(cv2.CAP_PROP_FRAME_COUNT))
                    # recording the detecting vedio 
                    print('frame total numer={0}, width={1}, Height={2}'.format(self.AgthmParams.totalNum_frame,self.AgthmParams.frame_width,self.AgthmParams.frame_height))
                    self.AgthmParams.video_image = np.zeros((self.AgthmParams.frame_height,self.AgthmParams.frame_width,3), np.uint8)
                    self.AgthmParams.video_counter = self.AgthmParams.totalNum_frame*2
                    
        except:
            # if not then just use OpenCV default
            print("INFO: camera_stream class not found - camera input may be buffered")
            self.AgthmParams.camera_l = cv2.VideoCapture(0)
            self.AgthmParams.camera_d = self.AgthmParams.camera_l

    def MainDeteTrack(self,orig_ocv,ori_depth):
        
        print('Initial NumofRows Setting = {0}'.format(self.LanesParams.FullDTHNumRowsInit)) 
        ########################################################################       

        self.AgthmParams.frame_num=self.AgthmParams.frame_num+1
        

        if self.AgthmParams.frame_num == self.AgthmParams.totalNum_frame-1:
            # recording_counter=recording_counter+1
            self.AgthmParams.camera_l = cv2.VideoCapture(self.AgthmParams.video_ocv)
            self.AgthmParams.camera_d = cv2.VideoCapture(self.AgthmParams.video_dpt)

        grabbed, ori_depth = self.AgthmParams.camera_d.read()
        grabbed_l, ori_ocv = self.AgthmParams.camera_l.read()

        while (self.AgthmParams.frame_num<200):
            self.AgthmParams.frame_num+=1
            grabbed, ori_depth = self.AgthmParams.camera_d.read()            
            continue

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
        # print('original dimension: ', ori_depth.shape)
        # snip section of video frame of interest & show on screen

        # row0 = 0
        # h0=375
        # col0 = 200
        # w0=300

        row0 =self.LanesParams.FullDptRow0 #= 0
        h0=self.LanesParams.FullDptH0#=375
        col0 =self.LanesParams.FullDptCol0# = 200
        w0=self.LanesParams.FullDptW0#=300
        snip = ori_depth[row0:row0+h0,col0:col0+w0]

        snpwidth = int(snip.shape[1])
        snip_l = ori_ocv[row0:row0+h0,col0:col0+w0]

        lanebandwith=self.LanesParams.FullDptbandwith    #30 #estimation or setting the initial lane wide
        Numoflanes = self.LanesParams.FullDTHNumRowsInit  #4
        laneoffset= self.LanesParams.FullDEPOffset  #25#25 #15  #25 
        LaneOriPoint = self.LanesParams.FullDEPLaneOriPoint   # 25 # where the row started

        lstartpt = LaneOriPoint + int (lanebandwith*0.5)
        lanegap=int((snpwidth-Numoflanes*lanebandwith-LaneOriPoint*2)/(Numoflanes-1))
        self.LanesParams.deltalane= lanebandwith+lanegap

        # lanegap=int((snpwidth-self.AgthmParams.FullRGBNumoflanes*self.AgthmParams.FullRGBbandwith-self.AgthmParams.FullRGBLaneOriPoint*2)/(self.AgthmParams.FullRGBNumoflanes-1))
        # height = int(snip.shape[0])
        # print('snap dimension: ', snip.shape)
        # plt.figure('Tracking...')
        # plt.title('Snap')
        # plt.imshow(snip,cmap='gray')
        # plt.show(block=True)
        # plt.show()

        # orig_frame_rsized = cv2.resize(orig_frame,(width,height))
        # listimages = [orig_frame]
        # listtitles = ["Original"]
        # listimages.append(snip)
        # listtitles.append('Snap')
        # plt.gcf().suptitle('snaped image witdth = ' + str(width) +
                            # '\nheight = ' + str(height))

             
        img_size = (snip.shape[1], snip.shape[0])
        m, m_inv = self.perspective_transform(snip,laneoffset)

        warped = cv2.warpPerspective(snip, m, img_size, flags=cv2.INTER_LINEAR) 
        warped_l = cv2.warpPerspective(snip_l, m, img_size, flags=cv2.INTER_NEAREST)
        cen_hight=(int)(0.382*warped.shape[0])
        # print('warped dimension: ', warped.shape)       
        # snip=np.copy(warped)
        # plt.figure('warping...')
        # plt.title('warped')
        # plt.imshow(snip, cmap='gray', vmin=0, vmax=1)
        # plt.show()

        # listimages.append(warped)
        # listtitles.append('warped')
        # plt.gcf().suptitle('warped image offset = ' + str(offset) +
                            # '\ncen_hight = ' + str(cen_hight))
        #################################                    
        # cv2.imwrite('/home/chfox/prototype/Weed3DJetPrj/ref_images/Fig55Lanewarpedrhm.jpg',warped)
        # plt.figure('warping...')
        # plt.title('warped')
        # plt.imshow(warped, cmap='gray', vmin=0, vmax=1)
        # plt.show()

        # inv_warped = cv2.warpPerspective(snip, m_inv, img_size, flags=cv2.INTER_LINEAR)
        # cv2.imwrite('/home/chfox/prototype/Weed3DJetPrj/ref_images/Fig55Lanein_warpedrhm.jpg',inv_warped)
        # plt.figure('in_warped...')
        # plt.title('in_warped')
        # plt.imshow(in_warped, cmap='gray', vmin=0, vmax=1)
        # plt.show()
        # res = np.hstack((inv_warped,warped)) #stacking images side-by-side
        # plt.figure('Paper writing')
        # plt.title('warped_in_warped') 
        # plt.imshow(res, cmap='gray', vmin=0, vmax=1)
        # cv2.imwrite('/home/chfox/prototype/Weed3DJetPrj/ref_images/Fig55warped_in_warpedrhm.jpg',res)
        # plt.show() 

        """
        https://jeffwen.com/2017/02/23/lane_finding

        """
        width = int(warped.shape[1] )
        height = int(warped.shape[0])
        """

        # create polygon (trapezoid) mask to select region of interest
        mask = np.zeros((snip.shape[0], snip.shape[1]), dtype="uint8")
        pts = np.array([[0, height], [0, 0], [width, 0], [width,height]], dtype=np.int32)
        # pts = np.array([[25, 190], [275, 50], [380, 50], [575, 190]], dtype=np.int32)
        cv2.fillConvexPoly(mask, pts, 255)
        # plt.figure('Tracking...')
        # plt.title('mask')
        # plt.imshow(mask,cmap='gray')
        # plt.show(block=True)
        # plt.show(block=False)
        # plt.pause(1)
        # plt.close()
        """
        masked = np.copy(warped)#cv2.bitwise_and(snip, snip, mask=mask)
        masked_l =np.copy(warped_l)
        # cv2.imshow("Region of Interest", masked)
        # cv2.waitKey(100)   
        # plt.figure('Tracking...')
        # plt.title('masked')
        # plt.imshow(masked,cmap='gray') 
        # plt.show(block=True)
        # plt.show()
        # plt.show(block=False)
        # plt.pause(1)
        # plt.close()
        bb,gg,rr = cv2.split(masked)
        # print('dimension of hsv_frame:', gg.shape)
        # Create our shapening kernel, it must equal to one eventually
        kernel_sharpening = np.array([[-1,-1,-1], 
                              [-1, 9,-1],
                              [-1,-1,-1]])

        # applying the sharpening kernel to the input image & displaying it.
        hsv_green_mask = cv2.filter2D(gg, -1, kernel_sharpening)
        # print('dimension of hsv_green_mask inRanged :', hsv_green_mask.shape)

        equ = cv2.equalizeHist(hsv_green_mask)
        # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        # cl1 = clahe.apply(hsv_green_mask)
        
        hsv_green_mask=equ

        #######################################
        # normalizedImg = cv2.normalize(hsv_green_mask,  hsv_green_mask, 0, 255, cv2.NORM_MINMAX)
        dims=5
        sgmcolor=75
        sgmspace=75
        blurred = cv2.bilateralFilter(hsv_green_mask,dims,sgmcolor,sgmspace) 

        rowk=27
        colk=27
        kernel = np.ones((rowk,colk),dtype=np.uint8)
        # Opening = erosion followed by dilation
        opening = cv2.morphologyEx(blurred, cv2.MORPH_OPEN, kernel)
        closing = cv2.morphologyEx(blurred, cv2.MORPH_CLOSE, kernel)


        thrimage = opening
        blcksiz = 21
        bordsize = 2
        maxval =200
        ret,th1 = cv2.threshold(thrimage,0,maxval,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        # listimages.append(th1)
        # listtitles.append('OTSU')

        th2 = cv2.adaptiveThreshold(thrimage,maxval,cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY,blcksiz,bordsize)        
        # listimages.append(th2)
        # listtitles.append('MEAN')

        th3 = cv2.adaptiveThreshold(thrimage,maxval,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,blcksiz,bordsize)
        # listimages.append(th3)
        # listtitles.append('GAUSSIAN')


         # convolute with proper kernels 
        ddepth = cv2.CV_16S#cv2.CV_64F#cv2.CV_16S
        kernel_size = 5


        sobelx = cv2.Sobel(th3,ddepth,1,0,ksize=kernel_size)  # x
        sobelx = cv2.convertScaleAbs(sobelx)


        sobely = cv2.Sobel(th3,ddepth,0,1,ksize=kernel_size)  # y
        sobely = cv2.convertScaleAbs(sobely)


        #######################################
        blurred = np.copy(sobelx)#laplacian#th3#th2#sobelx#th3
        # erosion = np.copy(blurred)#cv2.erode(blurred,kernel,iterations = 1) #thinning, the lines
        ###################################################
        # erosion = np.copy(sobelx)#cv2.erode(sobelx,kernel,iterations = 1) #thinning, the lines
        

        # green = cv2.bitwise_and(snip, snip, mask=erosion)
        # green_l = cv2.bitwise_and(snip_l, snip_l, mask=erosion)
        # res=np.hstack((green_l,green))
        # print('dimension of depth with mask=erosion :', green.shape)
        # plt.figure('Tracking...')
        # plt.title('Visibe and Depth Masked with Erosion')
        # plt.imshow(res,cmap='gray')
        # cv2.imwrite('./output/green.png',green)
        # plt.show(block=False)
        # plt.pause(0.5)
        # plt.close()

   
    
        # blurred=sobelx#th2#sobelx
        # blurred=closing
        minthres=50#100
        maxtres=200#150
        edges = cv2.Canny(blurred, minthres, maxtres) #150
        # plt.figure('Tracking...')
        # plt.title('closing')
        # plt.imshow(blurred,cmap='gray')   
        # plt.show(block=False)
        # plt.pause(1)
        # plt.close()
        # plt.show(block=True)
        # plt.show()

        # listimages.append(closing)
        # listtitles.append('morphology closing')
        # plt.gcf().suptitle('closing cols: ' + str(cols) +
                            # '\nsigma rows: ' + str(rows))     

       

        ####################################################
        # apply mask and on canny image on screen
        regionofintrest = edges#cv2.bitwise_and(edges, edges, mask=mask) # mask, mask0
        #get line segments from region of interest
        datax,datay,_lines_mid_v=self.get_lines(regionofintrest,masked)

        # cv2.imwrite('/home/chfox/prototype/Weed3DJetPrj/ref_images/Fig22LineSegsrhm.jpg',masked)
        # plt.figure('paper writing...')
        # plt.title('Line segments')
        # plt.imshow(masked,cmap='gray')
        # plt.show(block=True)
        # plt.show()

        #get lane sortd by finding the center of each lane
        cent_refined,masked,bandwith=self.get_lanes(datax,datay,_lines_mid_v,masked,cen_hight)      
        #get dictionary centroid <=> lane : (x,y)
        lane_dic= self.get_dic(cent_refined,_lines_mid_v,masked,bandwith)
    
        masked = self.draw_display(masked,lane_dic,cen_hight,width)
        if self.AgthmParams.ToIntializeSetting is True:
            masked_l = self.draw_display(snip_l,lane_dic,cen_hight,width)
            rows = int(masked.shape[0])
            for la in range(lstartpt,snpwidth,self.LanesParams.deltalane):                 
                self.AgthmParams.video_image = cv2.line(masked_l,(la,rows-1),(la,0),(0,0,0),10)
            return      
        
        ###########################################
        # dipmat=np.copy(snip)
        # draw_display(dipmat,lane_dic,cen_hight,width) 
        # cv2.imwrite('/home/chfox/prototype/Weed3DJetPrj/ref_images/Fig23LaneDetectionsniprhm.jpg',dipmat)
        # res=np.hstack((dipmat,masked))
        # plt.figure('paper writing...')
        # plt.title('Pure Lane detection ...')
        # plt.imshow(res,cmap='gray')
        # plt.show(block=True)
        # plt.show()


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
                
                # print('lane_centers = : ', lane_centers)      
                cx=(int)(key)
                cxy= np.median(lane_centers,axis=0)#
                track_pt=np.array((cxy[0][0],cxy[0][1]),np.float32)
                # print('track_pt = : ', track_pt) 
                x,y,w,h=(int)(cxy[0][0]),(int)(cxy[0][1]),10,10
                
                if self.AgthmParams.firstFrame is True:
                    # deltalane= self.LanesParams.FullRGBbandwith+lanegap
                    # deltalane= lanebandwith+lanegap
                    # print('Lane Gap = {0}, Lane Width = {1}, Total Lane width = {2}, loop Deta = {3}'.format(lanegap,self.AgthmParams.FullRGBbandwith,snpwidth,deltalane))
                    print('Lane Gap = {0}, Lane Width = {1}, Total Lane width = {2}, loop Deta = {3}'.format(lanegap,lanebandwith,snpwidth,self.LanesParams.deltalane))
                    NumofSettingID=0
                    # for la in range(self.LanesParams.FullRGBLaneOriPoint,snpwidth,deltalane):  
                    # for la in range(LaneOriPoint,snpwidth,deltalane): 
                    for la in range(lstartpt,snpwidth,self.LanesParams.deltalane):                    
                        track_pt=[la,cen_hight]
                        _lines_mid_v=[(la, cen_hight)]
                        self.AgthmParams.grasslanes[la]=grasstrack(la,masked,track_pt,(x,y,w,h),_lines_mid_v,lanebandwith)
                        NumofSettingID=NumofSettingID+1
                    print('total number of lanes tracked = {0}'.format(NumofSettingID))
                    # print('total number of lanes = {0}'.format(grasstrack.numofids))
                    self.AgthmParams.firstFrame=False

        if self.AgthmParams.firstFrame is False:
            # print('\n kalman tracking points ...........')
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
                    x,y,w,h=(int)(cxy[0][0]),(int)(cxy[0][1]),10,10
                    if abs(p.id-id)<(bandwith*0.618):                       
                       masked = p.update(masked,track_pt,(x,y,w,h))
                    #    snip = p.update(snip,track_pt,(x,y,w,h))
                       snip_l = p.update(snip_l,track_pt,(x,y,w,h))
                       p.id = int(0.5*(p.id+id))  # int(p.center[0])
                    #    unwarped = cv2.warpPerspective(masked, m_inv, (masked.shape[1], masked.shape[0]), flags=cv2.INTER_LINEAR)  # DEBUG
                    #    plt.figure('unwarping...')
                    #    plt.title('unwarped')
                    #    plt.imshow(unwarped, cmap='gray', vmin=0, vmax=1)
                    #    plt.show() 
                       found_flg=True
                       break
                if found_flg==False:
                    print("Lande id: %d is not updated!" % p.id)
                    width = int(masked.shape[1])
                    height = int(masked.shape[0])

                    pred_posx=int(p.prediction[0])
                    if abs(p.id-pred_posx)<(int)(bandwith*0.382):
                        # lane_mat.masked = cv.line(lane_mat.masked,(pred_posx,height-1),(pred_posx,0),(255,255,0),4)
                        masked = cv2.line(masked,(pred_posx,height-1),(pred_posx,0),(255,255,0),4)
                        # snip = cv2.line(snip,(pred_posx,height-1),(pred_posx,0),(255,255,0),4)
                        snip_l = cv2.line(snip_l,(pred_posx,height-1),(pred_posx,0),(255,255,0),4)
                        p.id=pred_posx
                    else:
                        # lane_mat.masked = cv.line(lane_mat.masked,(p.id,height-1),(p.id,0),(255,255,0),4)
                        # masked = cv.line(output.warped,(p.id,height-1),(p.id,0),color,4)
                        masked = cv2.line(masked,(int(p.id),height-1),(int(p.id),0),(255,255,0),4)
                        snip_l = cv2.line(snip_l,(int(p.id),height-1),(int(p.id),0),(255,255,0),4)
                        # snip = cv2.line(snip,(int(p.id),height-1),(int(p.id),0),(255,255,0),4)


        # cv2.imwrite('/home/chfox/prototype/Weed3DJetPrj/ref_images/Fig24TOPDownTrackrhm.jpg',masked)
        # cv2.imwrite('/home/chfox/prototype/Weed3DJetPrj/ref_images/Fig24TOPDownTracksniprhm.jpg',snip)
        # res = np.hstack((snip_l,masked)) #stacking images side-by-side
        # plt.figure('kalman tracking...')
        # plt.title('top-down tracks ...')
        # plt.imshow(res,cmap='gray')
        # plt.show(block=False)
        # plt.pause(0.5)
        # plt.show()

        unwarped = cv2.warpPerspective(masked, m_inv, (masked.shape[1], masked.shape[0]), flags=cv2.INTER_LINEAR)  # DEBUG
        unwarped_snip = cv2.warpPerspective(snip_l, m_inv, (snip_l.shape[1], snip_l.shape[0]), flags=cv2.INTER_LINEAR)  # DEBUG
        res = np.hstack((unwarped_snip,unwarped)) #stacking images side-by-side
        # cv2.imwrite('/home/chfox/prototype/Weed3DJetPrj/ref_images/Fig25BirdViewTrackrhm.jpg',unwarped)
        # cv2.imwrite('/home/chfox/prototype/Weed3DJetPrj/ref_images/Fig25BirdViewTracksniprhm.jpg',unwarped_snip)
        self.AgthmParams.video_image = cv2.resize(res,(self.AgthmParams.frame_width,self.AgthmParams.frame_height),interpolation=cv2.INTER_AREA)
        self.AgthmParams.out.write(self.AgthmParams.video_image)
        # plt.imshow(res, cmap='gray', vmin=0, vmax=1)
        # plt.show(block=False) 
        # plt.pause(0.25)      
        
        # plt.figure('kalman tracking...')
        # plt.title('bird-eyes tracks...')
        # plt.imshow(unwarped,cmap='gray')
        # plt.show(block=True)
        # plt.show()
        # key = cv2.waitKey(1)
        # plt.close('all') 
                
        # if key == 27:
        #     return
        
        # self.AgthmParams.out.release()
        # cv2.destroyAllWindows()





















"""



        
        # for pt in cent_refined:
        #     id=int(pt)
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
                for id in range(5,1000,195):  #5,200,195,580, 775,970 
                    track_pt=[id,cen_hight]
                    _lines_mid_v=[(id, cen_hight)]
                    self.AgthmParams.grasslanes[id]=grasstrack(id,masked,track_pt,(x,y,w,h),_lines_mid_v,bandwith)


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
                    print('lane_centers = : ', lane_centers)      
                    cx=(int)(key)
                    cxy= np.median(lane_centers,axis=0)#
                    track_pt=np.array((cxy[0][0],cxy[0][1]),np.float32)
                    print('track_pt = : ', track_pt) 
                    x,y,w,h=(int)(cxy[0][0]),(int)(cxy[0][1]),20,20
                    if abs(p.id-id)<(int)(bandwith*0.618):                       
                        masked = p.update(masked,track_pt,(x,y,w,h))
                        p.id=id#int(p.center[0])
                    
                    
                        #unwarped = cv2.warpPerspective(masked, m_inv, (masked.shape[1], masked.shape[0]), flags=cv2.INTER_LINEAR)  # DEBUG
                        plt.figure('kalman tracking...')
                        plt.title('track being updated')
                        plt.imshow(masked, cmap='gray', vmin=0, vmax=1)
                        plt.show(block=False)
                        plt.pause(0.25)
                        plt.close()
                        found_flg=True
                        break
                if found_flg==False:
                    print("Lande id: %d is not updated!" % p.id)
                    width = int(masked.shape[1])
                    height = int(masked.shape[0])
                    pred_posx=int(p.prediction[0])
                    # print("pred_posx={0:d}, pred_posy={1:d}".format(p.prediction[0],p.prediction[1]))
                    print("pred_posx= %d, pred_posy= %d" %(p.prediction[0],p.prediction[1]))
                    if abs(p.id-pred_posx)<(int)(bandwith*0.191):
                        masked = cv2.line(masked,(pred_posx,height-1),(pred_posx,0),(255,255,0),4)
                        p.id=pred_posx
                    else:
                        masked = cv2.line(masked,(p.id,height-1),(p.id,0),(255,255,0),4)

                    
                    plt.figure('kalman tracking...')
                    plt.title('track no updated')
                    plt.imshow(masked,cmap='gray')
                    plt.show(block=False)
                    plt.pause(0.25)
                    plt.close()

            unwarped = cv2.warpPerspective(masked, m_inv, (masked.shape[1], masked.shape[0]), flags=cv2.INTER_LINEAR)  # DEBUG
            plt.figure('kalman tracking...')
            plt.title('Lane Following')
            # cv2.resize(oriimg,None,fx=0.5,fy=0.5,interpolation=cv2.INTER_CUBIC)
            # img = cv2.resize(oriimg,(200,300),interpolation=cv2.INTER_AREA)
            self.AgthmParams.video_image = cv2.resize(unwarped,(self.AgthmParams.frame_width,self.AgthmParams.frame_height),interpolation=cv2.INTER_AREA)
            self.AgthmParams.out.write(self.AgthmParams.video_image)
            plt.imshow(self.AgthmParams.video_image, cmap='gray', vmin=0, vmax=1)
            plt.show(block=False) 
            plt.pause(0.25)                        
        ################################               
        ################################   
        """
        # key = cv2.waitKey(1)
        # plt.close('all') 
        
        # if key == 27:
        #     return
    
        # self.AgthmParams.out.release()
        # cv2.destroyAllWindows()

    # if __name__ == "__main__":
    # main()