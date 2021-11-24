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
import itertools

import os, glob
# from moviepy.editor import VideoFileClip
# built-in modules
import itertools as it
from tools.LaneTrackingKF import grasstrack
from tools.CSysParams import CLanesParams,CAlgrthmParams,CCamsParams
import time
# AgthmParams=object()
#https://divyanshushekhar.com/blob-detection-opencv-python/amp/
class CKeyptsBlobs():
    def __init__(self,fast_thresh=50,blob_thresh=225):
        # global AgthmParams
        # AgthmParams = abb()
        # self.AgthmParams = AgthmParams        
        self.cvfast_thresh = fast_thresh
        self.cvblob_thresh = blob_thresh

        self.font = cv2.FONT_HERSHEY_SIMPLEX
        # Create a detector with the parameters
        #https://github.com/spmallick/learnopencv/blob/master/BlobDetector/blob.py
        # Setup SimpleBlobDetector parameters.
        params = cv2.SimpleBlobDetector_Params()
        # Change thresholds
        params.minThreshold = 100
        params.maxThreshold = blob_thresh # 225
        
  

        params.filterByColor = True
        params.blobColor = 255

        # Filter by Area.
        params.filterByArea = True
        # params.minArea = 1
        params.maxArea =100

        # Filter by Circularity
        params.filterByCircularity = False
        params.minCircularity = 0.1

        # Filter by Convexity
        params.filterByConvexity = False
        params.minConvexity = 0.87
    
        # Filter by Inertia
        params.filterByInertia = False
        params.minInertiaRatio = 0.01       

        ##############################

        ver = (cv2.__version__).split('.')
        if int(ver[0]) < 3 :
	        self.detector = cv2.SimpleBlobDetector(params)
        else : 
	        self.detector = cv2.SimpleBlobDetector_create(params)
           
        # ORB......................
        self.orb = cv2.ORB_create(nfeatures=8000,scoreType=cv2.ORB_FAST_SCORE)  # OpenCV 3 backward incompatibility: Do not create a detector with `cv2.ORB()`.
        # FAST.....................
        self.fast = cv2.FastFeatureDetector_create(self.cvfast_thresh)#(40)
        # self.fast = cv2.FastFeatureDetector()  # Segmentation fault (core dumped)
        # self.fast = cv2.FeatureDetector_create('FAST')
        """
        # # surf
        self.sift =  cv2.xfeatures2d.SIFT_create() # patented 
        # SIFT
        self.surf =  cv2.xfeatures2d.SURF_create() # patented 
        """
       

    #lanes detection by meanshift/xmean clustering
    def get_cvBlobCentroids(self,ori_imag,hsv_frame, mask_gt):
       
        fig = plt.figure(num=None, figsize=(26, 12), dpi=80, facecolor='w', edgecolor='k')
        fig.canvas.set_window_title('Various Image Space')
        ax1 = fig.add_subplot(131)
        ax2 = fig.add_subplot(132)
        ax3 = fig.add_subplot(133)
        ax1.title.set_text('Warped to Origin')
        ax2.title.set_text('hsv_frame')
        ax3.title.set_text('mask_red_hsv')
        plt.subplot(1, 3, 1)
        plt.imshow(ori_imag, cmap="gray")
        plt.subplot(1, 3, 2)
        plt.imshow(hsv_frame, cmap="gray")
        plt.subplot(1, 3, 3)
        plt.imshow(mask_gt)
        plt.show(block=True)
        plt.close()
        ###############################################################
        ################################################################
        # red color boundaries [B, G, R]
        # https://bugsdb.com/_en/debug/0cb63884e8148260699146b26f14ee36
        lower = [1, 0, 20]
        upper = [60, 40, 200]

        lower =[17, 15, 100]
        upper =[50, 56, 200]
        
        # create NumPy arrays from the boundaries
        lower = np.array(lower, dtype="uint8")
        upper = np.array(upper, dtype="uint8")
        # find the colors within the specified boundaries and apply
        # the mask
        im_rgb = cv2.cvtColor(ori_imag, cv2.COLOR_BGR2RGB)
        mask = cv2.inRange(im_rgb, lower, upper)
        output = cv2.bitwise_and(im_rgb, im_rgb, mask=mask)
        ret,thresh = cv2.threshold(mask, 40, 255, 0)
        contours,hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if len(contours) != 0:
            # draw in blue the contours that were founded
            cv2.drawContours(output, contours, -1, 255, 3)
            #find the biggest area
            c = max(contours, key = cv2.contourArea)
            x,y,w,h = cv2.boundingRect(c)
            # draw the book contour (in green)
            cv2.rectangle(output,(x,y),(x+w,y+h),(0,255,0),2)

     
        fig = plt.figure(num=None, figsize=(26, 12), dpi=80, facecolor='w', edgecolor='k')
        fig.canvas.set_window_title('In RGB Space')
        ax1 = fig.add_subplot(221)
        ax2 = fig.add_subplot(222)
        ax3 = fig.add_subplot(223)
        ax4 = fig.add_subplot(224) 
        ax1.title.set_text('ori_imag')
        ax2.title.set_text('mask')
        ax3.title.set_text('thresh')
        ax4.title.set_text('output')
       
        plt.subplot(2,2, 1)
        plt.imshow(ori_imag, cmap="gray")
        plt.subplot(2, 2, 2)
        plt.imshow(mask, cmap="gray")
        plt.subplot(2, 2, 3)
        plt.imshow(thresh, cmap="gray")
        plt.subplot(2, 2, 4)
        plt.imshow(output, cmap="gray")
        plt.show()
        plt.close()

        ###############################################################
        ###############################################################
        # https://bugsdb.com/_en/debug/0cb63884e8148260699146b26f14ee36
        # find specific contours
        rows=3
        cols=3

        kernel = np.ones((rows,cols),dtype=np.uint8)
        lower_range = np.array([150, 100, 0])#np.array([150, 10, 10])
        upper_range = np.array([180, 255, 255])

        mask = cv2.inRange(hsv_frame, lower_range, upper_range)
        dilation = cv2.dilate(mask,kernel,iterations = 1)
        closing_gradient = cv2.morphologyEx(dilation, cv2.MORPH_GRADIENT, kernel)
        closing = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel)
        #Getting the edge of morphology
        edge = cv2.Canny(closing_gradient, 175, 175)
        res = np.hstack((mask,dilation)) #stacking images side-by-side
        fig = plt.figure(num=None, figsize=(26, 12), dpi=80, facecolor='w', edgecolor='k')
        fig.canvas.set_window_title('Filtering Image')
        ax1 = fig.add_subplot(221)
        ax2 = fig.add_subplot(222)
        ax3 = fig.add_subplot(223)
        ax4 = fig.add_subplot(224)
        ax1.title.set_text('mask & dilation')
        ax2.title.set_text('closing_gradient')
        ax3.title.set_text('closing')
        ax4.title.set_text('edge')
        plt.subplot(2,2, 1)
        plt.imshow(res, cmap="gray")
        plt.subplot(2, 2, 2)
        plt.imshow(closing_gradient)
        plt.subplot(2, 2, 3)
        plt.imshow(closing)
        plt.subplot(2, 2, 4)
        plt.imshow(edge)
        plt.show()
        plt.close()

      
        contours,hierarchy = cv2.findContours(edge, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # Find the index of the largest contour
        if len(contours) != 0:
        # draw in blue the contours that were founded
            cv2.drawContours(ori_imag, contours, -1, 125, 3)
            #find the biggest area
            c = max(contours, key = cv2.contourArea)
            x,y,w,h = cv2.boundingRect(c)
            # draw the book contour (in green)
            cv2.rectangle(ori_imag,(x,y),(x+w,y+h),(0,255,0),6)
            areas = [cv2.contourArea(c) for c in contours]
            max_index = np.argmax(areas)
            cnt=contours[max_index]
            x,y,w,h = cv2.boundingRect(cnt)
            cv2.rectangle(ori_imag,(x,y),(x+w,y+h),(255,0,0),2)

        plt.figure('using edge as input')
        plt.subplot(1, 3, 1)
        plt.imshow(ori_imag, cmap="gray")
        plt.subplot(1, 3, 2)
        plt.imshow(mask)
        plt.subplot(1, 3, 3)
        plt.imshow(edge)
        plt.show()
        plt.close()


        ####################################
        
        masked_red = cv2.bitwise_and(hsv_frame, hsv_frame, mask=mask_gt)
        plt.figure('bitwise for masked_red')
        plt.subplot(1, 3, 1)
        plt.imshow(hsv_frame, cmap="gray")
        plt.subplot(1, 3, 2)
        plt.imshow(mask_gt)
        plt.subplot(1, 3, 3)
        plt.imshow(masked_red)
        plt.show()
        plt.close()

        img = np.copy(mask_gt)

        ####################################
       

        #using contour detection:
        start = time.time()
        lenth = len(img.shape)  
        # create a binary thresholded image
        _, binary = cv2.threshold(img, 225, 255, cv2.THRESH_BINARY_INV)      
        # convert the grayscale image to binary image
        ret,thresh = cv2.threshold(img,127,255,0)
        res = np.hstack((binary,thresh))
        # show it
        plt.figure('centroids contour-favourite')
        plt.title('two thresholds')
        plt.imshow(res, cmap="gray")
        plt.show()

        ##################################################################

        # create a meshgrid for coordinate calculation
        r,c = np.shape(img)
        r_ = np.linspace(0,r,r+1)
        c_ = np.linspace(0,c,c+1)
        x_m, y_m = np.meshgrid(c_, r_, sparse=False, indexing='ij')
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        """
        https://docs.opencv.org/3.4/dd/d49/tutorial_py_contour_features.html

        Contour area is given by the function cv.contourArea() or from moments, M['m00'].
        area = cv.contourArea(cnt)

        Contour Perimeter
        It is also called arc length. 
        It can be found out using cv.arcLength() function. 
        Second argument specify whether shape is a closed contour (if passed True), or just a curve.

        perimeter = cv.arcLength(cnt,True)

        Contour Approximation
        To understand this, suppose you are trying to find a square in an image, but due to some problems in the image, you didn't get a perfect square, but a "bad shape" (As shown in first image below). Now you can use this function to approximate the shape. In this, second argument is called epsilon, which is maximum distance from contour to approximated contour. It is an accuracy parameter. A wise selection of epsilon is needed to get the correct output.

        epsilon = 0.1*cv.arcLength(cnt,True)
        approx = cv.approxPolyDP(cnt,epsilon,True)

        Convex Hull
        Convex Hull will look similar to contour approximation, but it is not (Both may provide same results in some cases). Here, cv.convexHull() function checks a curve for convexity defects and corrects it. Generally speaking, convex curves are the curves which are always bulged out, or at-least flat. And if it is bulged inside, it is called convexity defects. For example, check the below image of hand. Red line shows the convex hull of hand. The double-sided arrow marks shows the convexity defects, which are the local maximum deviations of hull from contours.
        hull = cv.convexHull(points[, hull[, clockwise[, returnPoints]]
        7.a. Straight Bounding Rectangle
        It is a straight rectangle, it doesn't consider the rotation of the object. So area of the bounding rectangle won't be minimum. It is found by the function cv.boundingRect().

        Let (x,y) be the top-left coordinate of the rectangle and (w,h) be its width and height.

        x,y,w,h = cv.boundingRect(cnt)
        cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        7.b. Rotated Rectangle
        Here, bounding rectangle is drawn with minimum area, so it considers the rotation also. The function used is cv.minAreaRect(). It returns a Box2D structure which contains following details - ( center (x,y), (width, height), angle of rotation ). But to draw this rectangle, we need 4 corners of the rectangle. It is obtained by the function cv.boxPoints()

        rect = cv.minAreaRect(cnt)
        box = cv.boxPoints(rect)
        box = np.int0(box)
        cv.drawContours(img,[box],0,(0,0,255),2)

        8. Minimum Enclosing Circle
        9. Fitting an Ellipse
        10. Fitting a Line
        Similarly we can fit a line to a set of points. Below image contains a set of white points. We can approximate a straight line to it.

        rows,cols = img.shape[:2]
        [vx,vy,x,y] = cv.fitLine(cnt, cv.DIST_L2,0,0.01,0.01)
        lefty = int((-x*vy/vx) + y)
        righty = int(((cols-x)*vy/vx)+y)
        cv.line(img,(cols-1,righty),(0,lefty),(0,255,0),2)

        """
        manaul_im = np.copy(ori_imag)
        for c in contours:
            # Get the boundingbox
            x,y,w,h = cv2.boundingRect(c)

            # calculate x,y coordinate of center
            # Get the corresponding roi for calculation
            weights = thresh[y:y+h,x:x+w]
            roi_grid_x = x_m[y:y+h,x:x+w]
            roi_grid_y = y_m[y:y+h,x:x+w]

            # get the weighted sum
            weighted_x = weights * roi_grid_x
            weighted_y = weights * roi_grid_y

            cx = int(np.sum(weighted_x) / np.sum(weights))
            cy = int (np.sum(roi_grid_y) / np.sum(weights))

            cv2.circle(img, (cx, cy), 5, 125, -1)  
            # draw the contour and center of the shape on the image
            cv2.drawContours(img, [c], -1, 255, 2)  
            cv2.putText(img, "centroid", (cx - 20, cy - 20),cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 2)
            
            cv2.circle(manaul_im, (cx, cy), 5, 255, -1)  
            # draw the contour and center of the shape on the image
            cv2.drawContours(manaul_im, [c], -1, (0, 255, 255), 2)  
            cv2.putText(manaul_im, "centroid", (cx - 20, cy - 20),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

            # blobCentrods.append([cx,cy])
            # display the image
            # cv2.imshow("Image", img)
            # cv2.waitKey(0)
            plt.figure('CvBlob-contour-self')
            plt.title('blobs-self-img')
            # cv2.imwrite('/home/dom/ARWAC/Documents/ARWAC/FarmLanedetection/ref_images/Fig11SobelxlineSegmts.jpg',masked)   
            plt.imshow(img,cmap='gray')
            plt.show(block=False)
            plt.pause(0.25)
            plt.close()

            plt.title('blobs-self-ori-imgag')

            plt.imshow(manaul_im,cmap='gray')
            plt.show(block=False)
            plt.pause(0.25)
            plt.close()


        ###################################################################    
        """       
        """        
        # find contours in the binary image
        contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        image = cv2.drawContours(ori_imag, contours, -1, (0, 255, 0), 2) 
        plt.figure('all in image')
        plt.title('blobs')
        # cv2.imwrite('/home/dom/ARWAC/Documents/ARWAC/FarmLanedetection/ref_images/Fig11SobelxlineSegmts.jpg',masked)   

        plt.imshow(image,cmap='gray')
        plt.show(block=True)
        plt.pause(0.25)
        plt.close()
        width = int(ori_imag.shape[1])
        height = int(ori_imag.shape[0])

        blobCentrods = list()
        for c in contours:# calculate moments for each contour
            M = cv2.moments(c)
            # calculate x,y coordinate of center
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])  
            else:
                 cX, cY = 0, 0

            # cX = int(M["m10"] / M["m00"])
            # cY = int(M["m01"] / M["m00"])
            ori_imag = cv2.line(ori_imag,(cX,height-1),(cX,0),(200,0,200),5)
            cv2.circle(ori_imag, (cX, cY), 5, (255, 255, 255), -1)  
            # draw the contour and center of the shape on the image
            cv2.drawContours(ori_imag, [c], -1, (0, 255, 0), 2)  
            cv2.putText(ori_imag, "centroid", (cX - 25, cY - 25),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            blobCentrods.append([cX,cY])
            # display the image
            # cv2.imshow("Image", img)
            # cv2.waitKey(0)
        plt.figure('CvBlob')
        plt.title('blobs')
        # cv2.imwrite('/home/dom/ARWAC/Documents/ARWAC/FarmLanedetection/ref_images/Fig11SobelxlineSegmts.jpg',masked)   

        plt.imshow(ori_imag,cmap='gray')
        plt.show(block=True)
        plt.pause(0.25)
        plt.close()

        elapsed = (time.time() - start)
        print('Centroids time: {:04.2f}'.format(elapsed))
        return blobCentrods

    def get_cvFASTKeypoints(self,img):
        """calculates centroid of a given matrix"""
        start = time.time()
        # height, width, channels = tuple(img.shape[1::-1])
        lenth = len(img.shape)
        if lenth >2:
            im = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        else:
            im = np.copy(img)
        # feataure matching
        # https://www.kaggle.com/wesamelshamy/tutorial-image-feature-extraction-and-matching
        #find and draw keypoints
        # https://answers.opencv.org/question/62595/python-cv2-fastfeaturedetector/
        # https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_fast/py_fast.html
        keypoints=self.fast.detect(im,None)
        img2 = cv2.drawKeypoints(img,keypoints, None,color=(0,255,0))
        # cv2.imwrite('/home/dom/ARWAC/awarc_lanedetect_statis/output_images/im_with_fastpt.jpg',img2)   


        T = [x.pt for x in keypoints]
        # datax=(int)[row[0] for row in T]
        # datay=[row[1] for row in T]
        datax=[row[0] for row in T]
        datay=[row[1] for row in T]


        # elapsed = (time.time() - start)
        # print('FAST time: {:04.2f}'.format(elapsed))
        """
        # plt.figure(figsize=(16, 16))
        plt.title('FAST Interest Points')
        plt.imshow(img2)
        plt.show(block=False)
        plt.pause(0.25)
        plt.close()
        #Print all default params
        print('Threshold: {}'.format(self.fast.getThreshold()))
        print('nomaxSuppression: {}'.format(self.fast.getNonmaxSuppression()))
        print('neighborhood: {}'.format(self.fast.getType()))
        print('Total keypoits with nonmaxSuppression: {}'.format(len(keypoints)))
        """
        # Disable nonmaxSuppression
        # self.fast.setBool('nonmaxSuppression',0)
        # kp = fast.detect(img,None)

        # #Disable nonmaxSuppression
        # self.fast.setNonmaxSuppresssion(0)
        # key_points = self.fast.detect(img, None)
        # print('Total keypoits without nonmaxSuppression: {}'.format(len(key_points)))
        # img3 = cv2.drawKeypoints(img, key_points, None, color=(0,0,255))
        # plt.figure(figsize=(16, 16))
        # plt.title('FAST Interest Points without NonmaxSupression')
        # plt.imshow(img3)
        # plt.show(block=True)
        # plt.pause(0.5)
        # plt.close()


        return keypoints , img2



    #lanes detection by meanshift/xmean clustering
    def get_cvBlobKeypoints(self,img):

        start = time.time()
        lenth = len(img.shape)
        if lenth >2:
            im = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        else:
            im = img.copy()
        #https://divyanshushekhar.com/blob-detection-opencv-python/amp/
        # Detect blobs from the image.
        keypoints = self.detector.detect(img)
        """
        lenth = len(keypoints)
        T = [x.pt for x in keypoints]


        newT = []
        key_f = lambda x: x[0]

        for key, group in itertools.groupby(T, key_f):
            if key<20:
                val =list(group)
                val=val.pop()
                newT.append(val)
                print(str(key) + ': ' + str(val))
        
        values = set(map(lambda x:x[1], T))
        newT = [[y[0] for y in T if y[1]==x] for x in values]

        datax=[row[0] for row in T]
        datay=[row[1] for row in T]
        """


        # datax = keypoints[:].pt[:,0]
        # datay = keypoints[:,1]
        # for i in range (lenth):
        #     point =keypoints[i].pt

        #     cx=(int)(point[0])
        #     cy=(int)(point[1])
        #     datax.append([cx])
        #     datay.append([cy])
            # cv2.circle(masked,(cx,cy),2,(0,255,255),-1) 

        # Draw detected blobs as red circles.
        # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS - This method draws detected blobs as red circles and ensures that the size of the circle corresponds to the size of the blob.
        # blobs = cv.drawKeypoints(img, keypoints, blank, (0,255,255), cv.DRAW_MATCHES_FLAGS_DEFAULT)
        im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        elapsed = (time.time() - start)
        print('cvBlob time: {:04.2f}'.format(elapsed))
        # Show blobs
        # cv2.imshow("Keypoints", im_with_keypoints)
        # cv2.waitKey(0)
        # Show keypoints
        plt.figure('CvBlob')
        plt.title('Keypoints') 
        cv2.imwrite('/home/dom/ARWAC/awarc_lanedetect_statis/output_images/im_with_keypoints.jpg',im_with_keypoints)   

        # cv2.imwrite('/home/dom/ARWAC/Documents/ARWAC/FarmLanedetection/ref_images/Fig11SobelxlineSegmts.jpg',masked)   
        plt.imshow(im_with_keypoints,cmap='gray')
        plt.show(block=False)
        plt.pause(0.25)
        plt.close()

        return keypoints,im_with_keypoints

   
    def get_cvORBKeypoints(self,img):
        """calculates centroid of a given matrix"""

        #https://blog.francium.tech/feature-detection-and-matching-with-opencv-5fd2394a590
        start = time.time()
        lenth = len(img.shape)
        if lenth >2:
            gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        else:
            gray_img = img.copy()
               #https://www.kaggle.com/wesamelshamy/tutorial-image-feature-extraction-and-matching
        # Detect blobs from the image only.        
        # find the keypoints with ORB
        kp = self.orb.detect(img,None)

         # both keypoints and descripter
        # kp, des = self.orb.detectAndCompute(gray_img, None)
        
        
        kp_img = cv2.drawKeypoints(img, kp, None, color=(0, 255, 0), flags=0)
        # kp_img = cv2.drawKeypoints(img, kp, None,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS) # Draw circles.
        
        elapsed = (time.time() - start)
        print('ORB time: {:04.2f}'.format(elapsed))
        plt.figure(figsize=(16, 16))
        plt.title('ORB Interest Points')
        plt.imshow(kp_img)
        plt.show(block=True)
        plt.pause(0.5)
        plt.close()

        return key_points

        start = time.time()
        img_building = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert from cv's BRG default color order to RGB
        #https://www.kaggle.com/wesamelshamy/tutorial-image-feature-extraction-and-matching
        # Detect blobs from the image.        
        key_points, description = self.orb.detectAndCompute(img_building, None)
        img_building_keypoints = cv2.drawKeypoints(img_building, key_points, img_building, 
                                           flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS) # Draw circles.
        
        elapsed = (time.time() - start)
        print('ORB time: {:04.2f}'.format(elapsed))
        plt.figure(figsize=(16, 16))
        plt.title('ORB Interest Points')
        plt.imshow(img_building_keypoints)
        plt.show(block=True)
        plt.pause(0.5)
        plt.close()

        return key_points


    def get_cvShi_TomasiCorners(self,img):
        """calculates centroid of a given matrix"""
        start = time.time()
        #find and draw keypoints
        # https://blog.francium.tech/feature-detection-and-matching-with-opencv-5fd2394a590
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        corners = cv2.goodFeaturesToTrack(gray_img, maxCorners=100,qualityLevel=0.02, minDistance=2)
        corners = np.float32(corners)

        for item in corners:
            x, y = item[0]
            cv2.circle(img, (x, y), 6, (0, 255, 0), -1)


        elapsed = (time.time() - start)
        print('good_features: {:04.2f}'.format(elapsed))
        plt.figure(figsize=(16, 16))
        plt.title('good_features')
        plt.imshow(img)
        plt.show(block=True)
        plt.pause(0.5)   

        return corners
    def get_cvHarriKeypoints(self,img):
          
        # https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_features_harris/py_features_harris.html        # Detect blobs from the image.
        lenth = len(img.shape)
        if lenth >2:
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()

        img_1 = img.copy()
        img_2 = img.copy()
        # find Harris corners
        width = int(img.shape[1] )
        height = int(img.shape[0])
        ############################   
        gray = np.float32(gray)
        # dst = cv2.cornerHarris(gray,2,3,0.04)
        dst = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)
        #result is dilated for marking the corners, not important
        dst = cv2.dilate(dst,None)
        # # Threshold for an optimal value, it may vary depending on the image.
        img_1[dst>0.01*dst.max()]=[255,0,0]
        # cv2.imshow('dst-before refining',img)
        # if cv2.waitKey(0) & 0xff == 27:
        #     cv2.destroyAllWindows()
        # plt.figure('Harris...')
        # plt.title('corners before refining')
        # # cv2.imwrite('/home/dom/ARWAC/Documents/ARWAC/FarmLanedetection/ref_images/Fig11SobelxlineSegmts.jpg',masked)   
        # plt.imshow(img,cmap='gray')
        # plt.show(block=False)
        # plt.pause(0.5)
        # plt.close()
        


        ret, dst = cv2.threshold(dst,0.01*dst.max(),255,0)
        dst = np.uint8(dst)
        # find centroids
        ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)

        # define the criteria to stop and refine the corners
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
        corners = cv2.cornerSubPix(gray,np.float32(centroids),(5,5),(-1,-1),criteria)


        # Now draw them
        res = np.hstack((centroids,corners))
        res = np.int0(res)

        len_pts = len(res)
        for id in range(len_pts):            
            if res[id][0]>=width:
                res[id][0]=width-1

            if res[id][1]>=height:
                res[id][1]=height-1

            if res[id][2]>=width:
                tm = res[id][2]
                res[id][2]=width-1
                tm = res[id][2]

            if res[id][3]>=height:
                res[id][3]=height-1
        
        img_2[res[:,1],res[:,0]]=[255,0,255]
        img_2[res[:,3],res[:,2]] = [0,255,0]

        cv2.imwrite('./output_images/subpixel5.png',img_2)

        results = np.hstack((img_1,img_2))
            
        plt.figure('Harris...')
        plt.title('corners before and after refining')
        # cv2.imwrite('/home/dom/ARWAC/Documents/ARWAC/FarmLanedetection/ref_images/Fig11SobelxlineSegmts.jpg',masked)   
        plt.imshow(results,cmap='gray')
        plt.show(block=True)
        plt.pause(0.5)
        plt.close()

        # cv2.imshow('dst',img)
        # if cv2.waitKey(0) & 0xff == 27:
        #     cv2.destroyAllWindows()
        return res
    def get_cvSURFFeatures(self,img):
        """calculates centroid of a given matrix"""
        start = time.time()
        #find and draw keypoints
        # https://blog.francium.tech/feature-detection-and-matching-with-opencv-5fd2394a590
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        kp, des = surf.detectAndCompute(gray_img, None)

        kp_img = cv2.drawKeypoints(img, kp, None, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        elapsed = (time.time() - start)
        print('SURF_features: {:04.2f}'.format(elapsed))
        plt.figure(figsize=(16, 16))
        plt.title('SURF_features')
        plt.imshow(kp_img)
        plt.show(block=True)
        plt.pause(0.5)   

        return corners
    def get_cvSIFTFeatures(self,img):
        """calculates centroid of a given matrix"""
        start = time.time()
        #find and draw keypoints
        # https://blog.francium.tech/feature-detection-and-matching-with-opencv-5fd2394a590
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        kp, des = sift.detectAndCompute(gray_img, None)

        kp_img = cv2.drawKeypoints(img, kp, None, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)


        elapsed = (time.time() - start)
        print('sift_features: {:04.2f}'.format(elapsed))
        plt.figure(figsize=(16, 16))
        plt.title('sift_features')
        plt.imshow(kp_img)
        plt.show(block=True)
        plt.pause(0.5)   

        return kp

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
            
    def MainDeteTrack(self,orig_ocv,ori_depth):
        
        print('Initial NumofRows Setting = {0}'.format(self.LanesParams.FullRRGNumRowsInit)) 
        ########################################################################       

        