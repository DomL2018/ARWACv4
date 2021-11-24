from __future__ import print_function
#lines_detection_hough_transform
#https://pysource.com/2018/03/07/lines-detection-with-hough-transform-opencv-3-4-with-python-3-tutorial-21/
"""
Faster video file FPS with cv2.VideoCapture and OpenCV
https://www.pyimagesearch.com/2017/02/06/faster-video-file-fps-with-cv2-videocapture-and-opencv/



further improvement:
Road Lanes Recognition With OpenCV, Python, and iOS.
https://medium.com/pharos-production/road-lane-recognition-with-opencv-and-ios-a892a3ab635c

"""
#Image being used
"""
https://stackoverflow.com/questions/42904509/opencv-kalman-filter-python

https://github.com/naokishibuya/car-finding-lane-lines/blob/master/Finding%20Lane%20Lines%20on%20the%20Road.ipynb

https://pysource.com/2019/02/15/detecting-colors-hsv-color-space-opencv-with-python/

Color Range in HSV 
# Red color
    low_red = np.array([161, 155, 84])
    high_red = np.array([179, 255, 255])
    red_mask = cv2.inRange(hsv_frame, low_red, high_red)
    red = cv2.bitwise_and(frame, frame, mask=red_mask)


# Blue color
    low_blue = np.array([94, 80, 2])
    high_blue = np.array([126, 255, 255])
    blue_mask = cv2.inRange(hsv_frame, low_blue, high_blue)
    blue = cv2.bitwise_and(frame, frame, mask=blue_mask)

# Green color
    low_green = np.array([25, 52, 72])
    high_green = np.array([102, 255, 255])
    green_mask = cv2.inRange(hsv_frame, low_green, high_green)
    green = cv2.bitwise_and(frame, frame, mask=green_mask)

# Every color except white
    low = np.array([0, 42, 0])
    high = np.array([179, 255, 255])
    mask = cv2.inRange(hsv_frame, low, high)
    result = cv2.bitwise_and(frame, frame, mask=mask)

"""
# Python 2/3 compatibility
print(__doc__)
import sys
PY3 = sys.version_info[0] == 3
import cv2
# import pyclustering
# from pyclustering.cluster import xmeans
import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
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
from moviepy.editor import VideoFileClip
# built-in modules
import itertools as it
from itertools import count
import operator
from csv import writer
from collections import OrderedDict
# parser = argparse.ArgumentParser()
# parser.add_argument("-a", "--algorithm",help = "m (or nothing) for meanShift and c for camshift")
# args = vars(parser.parse_args())
# get_curr_dir=os.getcwd()
# yolo_path = os.path.join(get_curr_dir,"gt_metrics")
# yolo_src_path = os.path.join(yolo_path,"src")
# yolo_utils_path = os.path.join(yolo_path, "Utils")
# tools_path = os.path.join(yolo_path, "tools")
# sys.path.append(yolo_path)
# sys.path.append(yolo_src_path)
# sys.path.append(yolo_utils_path)
# sys.path.append(tools_path)
# from points_extract.keypointsextract import CKeyptsBlobs
from tools.CSysParams import CLanesParams,CAlgrthmParams,CCamsParams#,CYoloV3Params
from tools.CInitialVideoSettingsGps import CVideoSrcSetting,CCameraParamsSetting,CRobtoParamSetting
# from gt_metrics.Detector_GT import gt_detect
from tools.HSVRGBSegt4HT import PltShowing,cvSaveImages,HSVFullEdgeProc, RGBColorSegmtFull, shadow_remove1#,shadow_remove2 #, 


from tools.svo_export import ParsingZED


grasslanes = {} # dictionary to hold the lane divisions by lnae1 , 2, 3, ...


parser = argparse.ArgumentParser(description='Perform ' + sys.argv[0] + ' example operation on incoming camera/video image')
parser.add_argument("-a", "--algorithm",help = "m (or nothing) for meanShift and c for camshift")
parser.add_argument("-c", "--camera_to_use", type=int, help="specify camera to use", default=0)
parser.add_argument("-r", "--rescale", type=float, help="rescale image by this factor", default=1.0)
parser.add_argument('video_file', metavar='video_file', type=str, nargs='?', help='specify optional video file')
args = parser.parse_args()

"""

lines = cv2.HoughLinesP(edges, 1, np.pi/180, 30, maxLineGap=250)
for line in lines:
    x1, y1, x2, y2 = line[0]
    cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
"""
def append_list_as_row(file_name, list_of_elem):
    # Open file in append mode
    with open(file_name, 'a+', newline='') as write_obj:
        # Create a writer object from csv module
        csv_writer = writer(write_obj)
        # Add contents of list as last row in the csv file
        csv_writer.writerow(list_of_elem)


def center(points):
    """calculates centroid of a given matrix"""
    x = (points[0][0] + points[1][0] + points[2][0] + points[3][0]) / 4
    y = (points[0][1] + points[1][1] + points[2][1] + points[3][1]) / 4
    return np.array([np.float32(x), np.float32(y)], np.float32)

font = cv2.FONT_HERSHEY_SIMPLEX
# The below is simlified with fixed lane position, currently, 2nd and 4th
def get_dic_gt2(gt_cents_v,LaneGap): 
    global LanesParams
    # gt_keys = LanesParams.Lane_GTs_keys
    # gt_value = LanesParams.Lane_GTs
    # lane_dic_gt, LanesParams.Lane_GTs=get_dic_gt2(LanesParams.Lane_GTs_keys, LanesParams.Lane_GTs,row_gt,bandwith)
    lenth_cents = len (gt_cents_v)   
    lane_dic_gt = {}
    # lane_coords_x = [] 
    # lane_coords_y = []  
    # row_gt2=sorted(row_gt, key=operator.itemgetter(1))   # Ascending order
    # print('row_gt2 = : ', row_gt2)
    # decended by 'True', asceended by 'False' along dimenstions [0], or [1]
    # row_gt=sorted(gt_cents_v, key=lambda x: x[0], reverse=True)
    # print('row_gt = : ', row_gt)
    gt_avlible = False
    row_gt=sorted(gt_cents_v, key=lambda x: x[0], reverse=False)
    print('ascending along x for row_gt = : ', row_gt) 
    if lenth_cents==2:
        # verify the groundtruth is right or not?
        gt1 = row_gt[0][0]
        gt2 = row_gt[1][0]
        if abs(gt2-gt1)>LaneGap: # it is true ground truth pegs detected

            lkeys = LanesParams.Lane_GTs_keys[1]
            index='lane_'+str(lkeys)
            lane_dic_gt[lkeys]=gt_cents_v[0]
            LanesParams.Lane_GTs[index]=gt1
            # meanx1 = np.mean(gt_cents_v[0],0)
            # meany1 = np.mean(gt_cents_v[0],1)

            lkeys = LanesParams.Lane_GTs_keys[2]
            index='lane_'+str(lkeys)#
            lane_dic_gt[lkeys]=gt_cents_v[1]            
            LanesParams.Lane_GTs[index]=gt2
            gt_avlible = True
            # meanx3 = np.mean(gt_cents_v[1],0)
            # meany3 = np.mean(gt_cents_v[1],1)        

    """
    indx = 0
    for cen in gt_keys: 
        indx+=1        
        for point in gt_cents_v: 
            if abs(cen-point[0])<bandwith:
                cx=point[0]
                cy=point[1]
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
    """
                if lkeys in lane_dic:
                    lane_dic_gt[lkeys].append((cx, cy)) 
                else:
                    lane_dic_gt[lkeys]=[(cx,cy)]
                # else: 
                #     Output[y] = [(x, y)] 
    """
    """
                lane_dic_gt[lkeys]=[meanx,meany]
                index='lane_'+str(lkeys)
                gt_value[index]=meanx
                # print('lenth of groun truth lane enteroids = ',len(lane_coords),len(lane_dic))
                # print(lane_dic)

    """
    return  lane_dic_gt, gt_avlible
# ground truth: holding the centroids from landmarks
def get_dic_gt(gt_keys, gt_value,gt_cents_v,bandwith):   
        
    lane_dic_gt = {}
    lane_coords_x = [] 
    lane_coords_y = [] 
    
    indx = 0
    for cen in gt_keys: 
        indx+=1        
        for point in gt_cents_v: 
            if abs(cen-point[0])<bandwith:
                cx=point[0]
                cy=point[1]
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
                lane_dic_gt[lkeys]=[meanx,meany]
                index='lane_'+str(lkeys)
                gt_value[index]=meanx
                # print('lenth of groun truth lane enteroids = ',len(lane_coords),len(lane_dic))
                # print(lane_dic)


    return  lane_dic_gt,gt_value
# calculate and merge multipple detection into regression 
def get_dic_centring(gt_keys, gt_cents_v,bandwith):   
        
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

    return  lane_dic_cent

def data_offset_analysis():
    datax=list()
    with open('./output/laneOffSet_file.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                print(f'1st Column is {", ".join(row)}')
                datax.append(row[0])
                line_count += 1
            else:
                print(f'\t{row[0]} in the row: {line_count} \n')
                datax.append(row[0])
                line_count += 1
        print(f'Processed {line_count} lines.')

    # datax=m_CLanParams.Lane_Ofset_Total
    widthx = len(datax)
    # heighty = len(datay)
    # print('widthx=', widthx)
    # print('heighty=', heighty)
    lenx = np.linspace(0, widthx, widthx)
    print('lenx=', widthx)
    # leny = np.linspace(0, heighty, heighty)
    pylab.scatter(lenx,datax)
    fig = pylab.gcf()
    fig.canvas.set_window_title('offset distribution...')
    plt.title('Off Set of Lane')
    pylab.scatter(lenx, datax)
    # pylab.savefig("./output/lincenters_a.png")
   
    pylab.show(block=False)        
    pylab.pause(0.25)
    pylab.close()

def centroid_refining_new(centroids,bandwith):
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

    while (inx < cen_lth):
        # if cen_lth==1:
        #     cur = centroids[inx]
        #     cent_refined.append(cur[inx])
        #     inx=inx+1
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
def centroid_refining(centroids,bandwith):

    cent_refined=[]    
    cen_lth = len(centroids)-1## all list index number
    inx=0 
       
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
#line segments detection by HoughP , 
def get_lines(regionofintrest,masked):    

        minLineLength = 4#30
        maxLineGap = 2#10
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
                if (abs(math.degrees(orient)) > 5.0) and abs(math.degrees(orient)) < (175.0):
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


        cvSaveImages(masked,'Fig11SobelxlineSegmts.jpg')
        PltShowing(masked,'Tracking...','line segments',True,True)


        datax=[]
        datay=[]
        if _lines_mid_v is not None:
            for point in _lines_mid_v:
                cx=(int)(point[0][0])
                cy=(int)(point[0][1])
                datax.append([cx])
                datay.append([cy])
                cv2.circle(masked,(cx,cy),2,(0,255,255),-1) 
        """
        widthx = len(datax)
        # heighty = len(datay)
        # print('widthx=', widthx)
        # print('heighty=', heighty)
        lenx = np.linspace(0, widthx, widthx)
        # leny = np.linspace(0, heighty, heighty)
        # pylab.scatter(lenx,datax)
        fig = pylab.gcf()
        fig.canvas.set_window_title('Tracking...')
        plt.title('center of lane')
        pylab.scatter(datax,datay)
        # pylab.scatter(lenx,datax)
        pylab.savefig("./output/lincenters.png")
        plt.show(block=False)
        plt.pause(0.25)
        # pylab.show()
        plt.close()
        """
        # plt.show(block=False)
        # plt.pause(1)
        # plt.close()
        #clustering
        return datax, datay,_lines_mid_v



#lanes detection by meanshift/xmean clustering
def get_lanes(datax,datay,_lines_mid_v,masked,cen_hight,bandwith): 

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
    print('bandwith_ested=',bandwith_ested)
    
    ms = MeanShift(bandwidth=bandwith, bin_seeding=True)
    ms.fit(datax)
    labels = ms.labels_
    centroids = ms.cluster_centers_
    print('lanebandwith = {0}, centroids= {1}'.format(bandwith,centroids))

    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)
    print("number of estimated clusters : %d" %n_clusters_)

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
        cent_refined = centroid_refining_new(cent_refined,bandwith)
        if lenofrefined_prv==len(cent_refined):
            break

    # print('centroids refined = ',cent_refined)
    print('lenth of centroids refined= ',len(cent_refined))
    if cent_refined is not None:
        for cent in cent_refined:
            print('cent value = ',cent)
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

def get_dic(cent_refined,_lines_mid_v,masked,bandwith):   
      
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


def draw_display(masked,lane_dic,cen_hight,width):
    
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
                masked = cv2.line(masked,(x,rows-1),(x,0),(200,0,200),5)
            # lefty =(int)(a+b*(cols-1))     #y=a+bx
            # masked = cv2.line(masked,(cols-1,lefty),(cx,cy),(255,0,255),8)
            else:
                # masked = cv2.line(masked,(cols-1,lefty),(0,righty),(200,255,200),8)
                # masked = cv2.line(masked,(cols-1,righty),(0,lefty),(200,255,200),8)
                masked = cv2.line(masked,(lefty,rows-1),(righty,0),(200,255,200),5)
        #draw_pathlines(masked, lane_dic, color=[255, 0, 0], thickness=8, make_copy=True)
        PltShowing(masked,'Display...','line regression',True,True)
        # for line in get_lines(lines):
        #     leftx, boty, rightx, topy = line
        #     cv2.line(masked, (leftx, boty), (rightx,topy), (0,0,255), 5) 
        # cv2.imshow("frame", masked)
        # cv2.waitKey(100)

"""grasstrack class
  each grasstrack is composed of a ROI, an ID and a Kalman filter
  so we create a grasstrack class to hold the object state
"""
class grasstrack():
  _ids = count(1)  
#   numofids=0
  # 计算给定ROI直方图，设置卡尔曼滤波器，并将其与对象的属性（self.kalman）关联
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
    self.roi = frame[y-h:y, x-w:x]#cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2HSV)
    # roi_hist = cv2.calcHist([self.roi], [0], None, [16], [0, 180])
    # self.roi_hist = cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

    # set up the kalman
    self.kalman = cv2.KalmanFilter(4,2)
    self.kalman.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]],np.float32)
    self.kalman.transitionMatrix = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]],np.float32)
    self.kalman.processNoiseCov = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],np.float32) * 0.05  #0.03
    self.measurement = np.array((2,1), np.float32) 
    self.prediction = np.zeros((2,1), np.float32)
    self.term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )
    self.track_pt=track_pt
    self.center = track_pt
    self.prediction = track_pt
    self.pre_updated_id = id
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
      img2 = cv2.rectangle(frame, (x,y), (x+w,y+h), (168,0,0),4)
     
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
    print("stracking state: x= %4.2f, and  y= %4.2f"%(self.center[0],self.center[1]))

    self.center=self.kalman.correct(self.center)
    width = int(frame.shape[1])
    height = int(frame.shape[0])
    print("state correction: x=%4.2f, and y= %4.2f "%(self.center[0], self.center[1]))
    tmx=self.center[0]
    tmy=self.center[1]

    if self.center[0]>width:
        tmx=width-4
    if self.center[1]>height:
        tmy=height-4


    cv2.circle(frame, (int(tmx), int(tmy)), 4, (0, 0,255), -1) 
    frame = cv2.line(frame,(int(tmx),height-1),(int(tmx),0),(255,255,0),4)
    # cv2.circle(frame, (int(self.center[0]), int(self.center[1])), 4, (0, 0,255), -1) 
    # frame = cv2.line(frame,(int(self.center[0]),height-1),(int(self.center[0]),0),(255,255,0),4)

    self.prediction = self.kalman.predict()      
    print("state prediction:x=%4.2f, and y= %4.2f "%(self.prediction[0],self.prediction[1]))  

    # cv2.circle(frame, (int(prediction[0]), int(prediction[1])), 4, (255, 0, 0), -1)
    # displaying information on top left 
    # cv2.putText(frame, "ID: %d -> %s" % (self.id, self.center), (11, (self.id + 1) * 25 + 1),
    #     font, 0.6,
    #     (255, 0, 0),
    #     1,
    #     cv2.LINE_AA)
    # # indicating the information on top left
    # cv2.putText(frame, "ID: %d -> %s" % (self.id, self.center), (10, (self.id + 1) * 25),
    #     font, 0.6,
    #     (255, 255, 0),
    #     1,
    #     cv2.LINE_AA)

    PltShowing(frame,'kalman filter','tracking', False,False)
    return frame


def perspective_transform(img,offset):
	
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
    
    # warped = cv2.warpPerspective(img, m, img_size, flags=cv2.INTER_LINEAR)
    # unwarped = cv2.warpPerspective(warped, m_inv, (warped.shape[1], warped.shape[0]), flags=cv2.INTER_LINEAR)  # DEBUG
    # plt.figure('warping...')
    # plt.title('warped')
    # plt.imshow(warped, cmap='gray', vmin=0, vmax=1)
    # plt.show()

    # plt.figure('unwarping...')
    # plt.title('unwarped')
    # plt.imshow(unwarped, cmap='gray', vmin=0, vmax=1)
    # plt.show()

    


    """
    #full image
    # pts = np.array([[0, height], [0, 0], [width, 0], [width,height]], dtype=np.int32)
    # pts = np.array([[25, 190], [275, 50], [380, 50], [575, 190]], dtype=np.int32)

    cv2.fillConvexPoly(mask0, pts, 255)
    # cv2.imshow("Mask", mask)
    # cv2.waitKey(100)
    plt.figure('Tracking...')
    plt.title('mask0')
    plt.imshow(mask0,cmap='gray')
    plt.show(block=True)
    plt.show()
    
    return warped, unwarped, m, m_inv
	# src = np.float32(
	# 	[[200, 720],
	# 	[1100, 720],
	# 	[595, 450],
	# 	[685, 450]])
	# dst = np.float32(
	# 	[[300, 720],
	# 	[980, 720],
	# 	[300, 0],
	# 	[980, 0]])
    """


def main():
    """   
    """
    global grasslanes,LanesParams
    gt_avlible = False  # control the comparsion with groundtruth
    fields = ['lane_1', 'lane_3']
    errfilename = '/media/dom/Elements/data/data_collection/data26112020/hstdata.csv'
    numofGt =0 # counter of number of ground truth detection
    # Set up the blob detector.   
    LanesParams=CLanesParams() 
    # YoloSetParams = CYoloV3Params() 
    # m_detector = CKeyptsBlobs()
    m_VideoSrcSet = CVideoSrcSetting()
    #topdown
    ###################################################################################
    # src_vid = m_VideoSrcSet.data_07082020_5deg_new_720_1()
    # src_vid = m_VideoSrcSet.data_18082020_85deg_720q()
    # src_vid = m_VideoSrcSet.data_02092020_deg_all()
    # src_vid = m_VideoSrcSet.data_09092020_deg_all()
    # src_vid = m_VideoSrcSet.data_22102020_deg_all()

    # src_vid = m_VideoSrcSet.data_05112020_4lanes_all()
    # src_vid = m_VideoSrcSet.data_12112020_4lanes_all()
    # src_vid = m_VideoSrcSet.data_19112020_4lanes_all()
    src_vid = m_VideoSrcSet.data_26112020_4lanes_all()
    # src_vid = m_VideoSrcSet.data_05012021_svo_all()


    campitch = src_vid['pitch']
    video_fille_l =   src_vid['path']#="/media/dom/Elements/data/TX2_Risehm/28072020/1st/goVL720q.avi"
    print("Reading file: {0}".format(video_fille_l))

    LaneOriPoint = src_vid['LaneOriPointL']#=50
    LaneEndPoint = src_vid['LaneEndPointR']#=50
    lanebandwith= src_vid['lanebandwidth']#=80

    Numoflanes = src_vid['NumofLanes']#=6
    laneoffset= src_vid['laneoffset']#=155
    lanegap=int((LaneEndPoint-LaneOriPoint)/(Numoflanes-1))   

    laneStartPo = int(lanegap*0.618)
    if laneStartPo > int(lanebandwith*0.618):
        laneStartPo = int(lanebandwith*0.618) 
    # due  to the wide lane in this time
    laneStartPo = int(lanegap*0.382)
    print("laneStartPo : {0}".format(laneStartPo)) 
    

    row0 =  src_vid['row0']#=350 image cropping top point
    col0 = int(LaneOriPoint-laneStartPo) # src_vid['col0']#=175  image cropping left point

    w0= int (LaneEndPoint-LaneOriPoint+laneStartPo*2)# int(src_vid['w0'])#=1025 total width of cropping image
    h0 = src_vid['frame_height'] # t   
    frm2strart = src_vid['frm2start']#=980
    print ('size of window row = {0}, width = {1} '.format(h0,w0))

    avi_width =  w0#src_vid['frame_width']#=1025
    avi_height = h0#src_vid['frame_height']#=370    

    # Split filename and extension.    
    head, tail = ntpath.split(video_fille_l)
    (name_pref, name_ext) = os.path.splitext(tail)
    upd_thrsh=20 # critirial for using kalman updting
    klman_start=0# control of updating 'id' with kalman filter >30: #
    # store errors 
    TotalError1 = [] # coresponding to second lane in this case
    TotalError3 = [] # coresponding to third lane in this case

    try:
        # to use a non-buffered camera stream (via a separate thread)
        if not(video_fille_l):
            import camera_stream
            camera = camera_stream.CameraVideoStream()
        else:
            # Check if camera opened successfully            
            camera = cv2.VideoCapture(video_fille_l) 
            
            # Find the number of frames
            clip = VideoFileClip(video_fille_l)
            time_length = clip.duration
            print( 'time_length',time_length)            
            
            frame_seq = int(camera.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
            print ("Number of frames: ", frame_seq)
            fps = camera.get(cv2.CAP_PROP_FPS)
            print("fps = %4.2f"%fps)

            frame_no = ((frame_seq) /(time_length*fps))
            print("current frame = %4d"%frame_no)
            if (camera.isOpened()== False): 
                print("Error opening video stream or file")
                # cap = cv2.VideoCapture() # not needed for video files
                
    except:
        # if not then just use OpenCV default
        print("INFO: camera stream class not found - camera input may be buffered")
        # camera = cv2.VideoCapture(0)
        exit()
        
    # 创建主显示窗口
    # cv2.namedWindow("Tracking...")
    # 设置任性字典和firstFrame标志（该标志使得背景分割器能利用这些帧来构建历史）
    
    firstFrame = True
    frame_num = -1
    vis_size = (avi_width,avi_height)
    fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    # out1 = cv2.VideoWriter('./output_images/Wheatoutpy1.mp4',fourcc1, 20, (avi_width,avi_height))
    svRes = '/media/dom/Elements/data/data_collection/data26112020/'    
    svRctRes = svRes+'KHRes'+campitch+'.avi'
    outRes = cv2.VideoWriter(svRctRes,fourcc, 10.0,vis_size)
    svRctOri = svRes+'KHOri'+campitch+'.avi'
    outOri = cv2.VideoWriter(svRctOri,fourcc, 10.0,vis_size)
 
    print('outRes dimension: ', vis_size)   
    #一行一行地读取视频帧
    grasslanes = {}
    lane_regs={} # keep the lane's name holder , according to the lanes initialization at begining

    while (camera.isOpened()):        
        # ret, orig_frame = video.read()
        frame_num=frame_num+1
        grabbed, orig_frame = camera.read()
        if (grabbed is False):
            print ("failed to grab frame.")
            camera = cv2.VideoCapture(0)
            continue

        if frame_num<=frm2strart:
            continue

        if frame_num%1!=0:
            continue

        klman_start +=1
        # imgcopy = np.ndarray.copy(orig_frame)    
        print( "\n -------------------- FRAME Count %d --------------------" % klman_start)    
        print('origin dimension: ', orig_frame.shape)   
        width_ori = int(orig_frame.shape[1] )
        height_ori = int(orig_frame.shape[0])  


        m, m_inv = perspective_transform(orig_frame,laneoffset)
        # cvSaveImages(orig_frame,'FigHistWheatOri0511.jpg')
        img_size = (width_ori, height_ori)
        ori_warped = cv2.warpPerspective(orig_frame, m, img_size, flags=cv2.INTER_LINEAR)
        report = np.hstack((ori_warped, orig_frame)) #stacking images side-by-side
        # Display the left image from the numpy array
        # cv2.imshow("Image", image_ocv)
        # plt.figure('ZED Camera Live')
        # plt.title('depth_zed')        
        # plt.imshow(report,cmap='gray')
        # plt.show(block=False)
        # plt.pause(0.05)
        # plt.close()
        PltShowing(report,'Camera Output...','Origin', True,True)




        # testing on whole image

        snip = orig_frame[row0:row0+h0,col0:col0+w0]
        # PltShowing(snip,'Tracking...','snap', True,True)
        # cvSaveImages(snip,'FigHistWheatCrop2010.jpg')
        pth = 'WheatSnip'+ campitch
        pth = pth +'.jpg'
        # snip section of video frame of interest & show on screen 
        snpwidth = int(snip.shape[1])

        # orig_frame_rsized = cv2.resize(orig_frame,(width,height))
        # listimages = [orig_frame]
        # listtitles = ["Original"]
        # listimages.append(snip)
        # listtitles.append('Snap')
        # plt.gcf().suptitle('snaped image witdth = ' + str(width) +
                            # '\nheight = ' + str(height))
             
        img_size = (snip.shape[1], snip.shape[0])
        # print('img_size dimension: ', img_size)  
        # m, m_inv = perspective_transform(snip,laneoffset)

        warped = ori_warped[row0:row0+h0,col0:col0+w0]
        # warped = cv2.warpPerspective(snip, m, img_size, flags=cv2.INTER_LINEAR) 
        cen_hight=(int)(0.382*warped.shape[0])
        report = np.hstack((warped, snip)) #stacking images side-by-side
        # Display the left image from the numpy array
        # cv2.imshow("Image", image_ocv)
        # plt.figure('ZED Camera Live')
        # plt.title('depth_zed')        
        # plt.imshow(report,cmap='gray')
        # plt.show(block=False)
        # plt.pause(0.05)
        # plt.close()
        PltShowing(report,'Cropped Output...','warped-Snip', True,True)
        # print('warped dimension: ', warped.shape)    
        #############################  
         #call gt detection for the landmark of lane ground truth
        # prediction, image, detect =gt_detect(name_pref,YoloSetParams,warped,frame_num)
        # detect = np.copy(cv2.cvtColor(warped,cv2.COLOR_BGR2RGB))
        detect = np.copy(warped)
        cols = int(warped.shape[1] )
        rows = int(warped.shape[0])
      
        ##############################################################
        width = int(warped.shape[1] )
        height = int(warped.shape[0])
        # create polygon (trapezoid) mask to select region of interest
        mask = np.zeros((snip.shape[0], snip.shape[1]), dtype="uint8")
        pts = np.array([[0, height], [0, 0], [width, 0], [width,height]], dtype=np.int32)
        # pts = np.array([[25, 190], [275, 50], [380, 50], [575, 190]], dtype=np.int32)
        cv2.fillConvexPoly(mask, pts, 255)
        """
        image = cv2.bitwise_and(orig_frame, orig_frame, mask=mask)
        # stacked_mask_3d = np.stack((mask,)*3, axis=-1)
        # mask =mask.astype(np.uint8)
        plt.figure('Tracking...')
        plt.title('mask image')
        plt.imshow(image,cmap='gray')
        plt.show(block=True)   
        """
        masked = np.copy(warped)#cv2.bitwise_and(snip, snip, mask=mask)
        hsv_frame = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
        # listimages.append(hsv_frame)
        # listtitles.append('hsv space')
        # inRange(hue, 42, 106, greenMask)
        # Green color
        # low_green = np.array([50, 60, 60])
        # high_green = np.array([80, 255, 255])        
        # low_green = np.array([42, 25, 25])
        # high_green = np.array([106, 225, 225])
        # low_green = np.array([25, 52, 72])
        # high_green = np.array([102, 255, 255])
        # low_yellow = np.array([18, 94, 140])
        # up_yellow = np.array([48, 255, 255])
        # mask = cv2.inRange(hsv_frame, low_green, up_green)

        low_green =src_vid['low_green']
        high_green = src_vid['high_green'] 
        regionofintrest = HSVFullEdgeProc(warped,low_green,high_green,name_pref) 
        # green_wheat,regionofintrest = RGBColorSegmtFull(warped,name_pref)        
        # hsv_green_mask = cv2.inRange(hsv_frame, low_green, high_green)

        ####################################################
        # apply mask and on canny image on screen
        # regionofintrest = cv2.bitwise_and(edges, edges, mask=mask) # mask, mask0
        #get line segments from region of interest
        datax,datay,_lines_mid_v=get_lines(regionofintrest,masked)

        #get lane sortd by finding the center of each lane
        cent_refined,masked,bandwith=get_lanes(datax,datay,_lines_mid_v,masked,cen_hight,lanebandwith)      
        #get dictionary centroid <=> lane : (x,y)
        lane_dic= get_dic(cent_refined,_lines_mid_v,masked,lanebandwith)
    
        draw_display(masked,lane_dic,cen_hight,width)
        
        ###########################################
         # for pt in cent_refined:
        if firstFrame is True:
            # deltalane= lanebandwith+lanegap
            deltalane= lanegap
            print('Lane Gap = {0}, Lane Width = {1}, Total Lane width = {2}, loop Deta = {3}'.format(lanegap,lanebandwith,snpwidth,deltalane))
            NumofSettingID=0

            # for la in range(LaneOriPoint,snpwidth,deltalane):  
            for la in range(laneStartPo,snpwidth-laneStartPo+1,deltalane):           
                track_pt=[la,cen_hight]
                _lines_mid_v=[]#[(la, cen_hight)]
                x,y,w,h=la,cen_hight,10,10
                grasslanes[la]=grasstrack(la,masked,track_pt,(x,y,w,h),_lines_mid_v,lanebandwith,deltalane)
                index='lane_'+str(la)
                # lane_regis[index] = grasslanes[la] # just for the registration of identification convenient
                NumofSettingID=NumofSettingID+1
                #temporay testing
                # index='lane'+str(NumofSettingID)
                LanesParams.Lane_GTs[index]=la
                LanesParams.Lane_GTs_keys.append(la)
                # lanes_gt=get_dic_gt(gt_keys, blobCentrods,bandwith)

            print('total number of lanes = {0}'.format(NumofSettingID))
                # print('total number of lanes = {0}'.format(grasstrack.numofids))
            firstFrame=False

        width = int(masked.shape[1])
        height = int(masked.shape[0])
        # lane_dic_gt, LanesParams.Lane_GTs=get_dic_gt(LanesParams.Lane_GTs_keys, LanesParams.Lane_GTs,row_gt,bandwith)
        LanesParams.Lane_Detect_val={}
        if firstFrame is False:
            print('\n kalman tracking frame ...........',klman_start)
            for ikey, p in grasslanes.items():#.iteritems(): 
                found_flg=False # tracking the updated lanes                      
                # keep last step id
                pre_id = p.id
                pre_predic_id = p.pre_updated_id
                masked = cv2.line(masked,(ikey,height-1),(ikey,0),(0,0,0),2)
                # p.prediction[0] = p.id
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
                    cxy= np.median(lane_centers,axis=0)#
                    track_pt=np.array((cxy[0][0],cxy[0][1]),np.float32)
                    print('track_pt = : ', track_pt) 
                    x,y,w,h=(int)(cxy[0][0]),(int)(cxy[0][1]),10,10
                    if abs(ikey-key)<lanegap*0.618:#(lanebandwith*1.618):                       
                       masked = p.update(masked,track_pt,(x,y,w,h))                       
                       p.id=ikey*0.382+key*0.618 # must keep here before later applying kalman filter
                    #    p.id=key  # for the wide lane , trust detection
                       if klman_start>=upd_thrsh: #abs(p.id-p.center[0])<(lanebandwith*0.1545):
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
                    if klman_start>upd_thrsh:
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
                LanesParams.Lane_Detect_val[index]=p.id
                # for the practical offset to control the robot manoveoveor
                LanesParams.Ctrl_Ofset[index]=pre_id-p.id
                # output in mm in evrage of all lanes
                LanesParams.Ctrl_Ofset[index]=LanesParams.Ctrl_Ofset[index]*LanesParams.PixelSize
                

        if gt_avlible == True:
            numofGt = numofGt+1
            lkeys = LanesParams.Lane_GTs_keys[1]
            index='lane_'+str(lkeys)
            # grounth inpectation with detection
            Detec_Err1=abs(LanesParams.Lane_GTs[index]-LanesParams.Lane_Detect_val[index])
            LanesParams.Detec_Err[index]=TotalError1.append(Detec_Err1)
            print('mean of lane_1 error = :',sum(TotalError1)/len(TotalError1))
            # for the third lane : from left to right
            lkeys = LanesParams.Lane_GTs_keys[2]
            index='lane_'+str(lkeys)#
            # grounth inpectation with detection
            Detec_Err3=abs(LanesParams.Lane_GTs[index]-LanesParams.Lane_Detect_val[index])
            LanesParams.Detec_Err[index]=TotalError3.append(Detec_Err3)
            print('mean of lane_3 error = :',sum(TotalError3)/len(TotalError3))
            # writing to csv file  
            list_of_elem = [Detec_Err1,Detec_Err3]
            append_list_as_row(errfilename, list_of_elem)


        # cvSaveImages(masked,'Fig24TOPDownTrack.jpg')
        # cvSaveImages(detect,'Fig24TOPDownTracksnip.jpg')        
        res = np.hstack((detect,masked)) #stacking images side-by-side
        PltShowing(res,'kalman tracking...','top-down tracks',False, False)

        unwarped = cv2.warpPerspective(masked, m_inv, (masked.shape[1], masked.shape[0]), flags=cv2.INTER_LINEAR)  # DEBUG
        unwarped_snip = cv2.warpPerspective(detect, m_inv, (snip.shape[1], snip.shape[0]), flags=cv2.INTER_LINEAR)  # DEBUG
        res = np.hstack((unwarped_snip,unwarped)) #stacking images side-by-side
        
        # cvSaveImages(unwarped,'Fig25BirdViewTrack.jpg')
        # cvSaveImages(unwarped_snip,'Fig25BirdViewTracksnip.jpg')
        PltShowing(res,'kalman tracking...','Perspective View Tracks',True, True)

        # resize image
        avi_res = cv2.resize(cv2.cvtColor(unwarped_snip, cv2.COLOR_BGR2RGB) , (avi_width,avi_height), interpolation = cv2.INTER_AREA)
        outRes.write(avi_res)
        ################################   
        """
                width = int(lane_mat.masked.shape[1])
                height = int(lane_mat.masked.shape[0])
                pred_posx=int(p.prediction[0])
                # print("pred_posx={0:d}, pred_posy={1:d}".format(p.prediction[0],p.prediction[1]))
                # print("pred_posx= %d, pred_posy= %d" %(p.prediction[0],p.prediction[1]))
                if abs(p.id-pred_posx)<(int)(lane_mat.bandwith*0.191):
                    # lane_mat.masked = cv.line(lane_mat.masked,(pred_posx,height-1),(pred_posx,0),(255,255,0),4)
                    lane_mat.masked = cv.line(output.warped,(pred_posx,height-1),(pred_posx,0),color,4)
                    p.id=pred_posx
                else:
                    # lane_mat.masked = cv.line(lane_mat.masked,(p.id,height-1),(p.id,0),(255,255,0),4)
                    lane_mat.masked = cv.line(output.warped,(p.id,height-1),(p.id,0),color,4)
        """
        ###################################       

        key = cv2.waitKey(1)
        plt.close('all') 
        
        if key == 27:
            break
       
    outRes.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
  main()
