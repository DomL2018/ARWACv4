#This file contains the python implementation of shadow detector for satellite imagery
#Author: Bhavan Vasu
# from skimage import io, color
import matplotlib.pyplot as plt
from PIL import  Image
#############
# LIBRARIES
#############
import numpy as np
import cv2
import os
import sys
import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.image as mpimg
import scipy
from scipy.optimize import leastsq
from scipy.stats.mstats import gmean
from scipy.signal import argrelextrema
from scipy.stats import entropy
from scipy.signal import savgol_filter
import statistics
import math
import time
import pylab as pl
import random
# https://github.com/V2dha/Color-Detection-Using-OpenCV-Python/blob/master/Color%20Detection.py
def PltShowing(img,figname,titlname, pauseflg=False,disflg=True):
    # return
    if disflg == True:
        plt.figure(figname,figsize=(36,24))
        plt.title(titlname)
        # plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB),cmap='gray')
        plt.imshow(img,cmap='gray')
        plt.show(block=pauseflg)  
        plt.pause(0.5)
        plt.close()

def cvSaveImages(img,figname='xx.jpg',path='./', svflg=False, subpth = '',extname='.jpg'):
    if svflg==True:
        nampepath = os.path.join(path, subpth, figname) 
        cv2.imwrite(nampepath,cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

def RGBColorSegmt(src):
    # https://blog.csdn.net/weixin_44524040/article/details/90350701
    # using RGB to do sigmenation
    # 使用2g-r-b分离土壤与背景
    t = time.time()
    # src = cv2.imread('C:\\Users\\zjk\\PycharmProjects\\untitled1\\1.bmp')
    # cv2.imshow('src', src)

    # 转换为浮点数进行计算
    fsrc = np.array(src, dtype=np.float32) / 255.0
    (b, g, r) = cv2.split(fsrc)
    gray = 2 * g - b - r#由于一张图片最大255，如果我2*g - b -r 超出了255 默认是255，但是如果是小数，怎么也不会超出255,因为uint8类型超过255就是这个数减255截断

    # 求取最大值和最小值
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray)
    """
    # 计算直方图
    # hist = cv2.calcHist([gray], [0], None, [256], [minVal, maxVal])
    # plt.plot(hist)
    # plt.show()

    height,width,channels = src.shape
    b,g,r = cv2.split(hsv_frame)
    rgb_split = np.empty([height,width*3,3],'uint8')
    rgb_split[:, 0:width] = cv2.merge([b,b,b])
    rgb_split[:, width:width*2] = cv2.merge([g,g,g])
    rgb_split[:, width*2:width*3] = cv2.merge([r,r,r])

    plt.title('Split BGR')
    plt.imshow(rgb_split)
    plt.show()
    # cv2.imshow("Channels",rgb_split)
    # cv2.moveWindow("Channels",0,height)

    plt.title('Split HSV')
    h,s,v = cv2.split(hsv_frame)
    hsv_split = np.concatenate((h,s,v),axis=1)
    plt.imshow(hsv_split)
    plt.show()
    """
    # 转换为u8类型，进行otsu二值化
    gray_u8 = np.array((gray - minVal) / (maxVal - minVal) * 255, dtype=np.uint8)
    gray_u8 = cv2.equalizeHist(gray_u8)
    (thresh, bin_img) = cv2.threshold(gray_u8, -1.0, 255, cv2.THRESH_OTSU)
    # cv2.imshow('bin_img', bin_img)
    ######################################

    dims=5
    sgmcolor=75
    sgmspace=75
    blurred = cv2.bilateralFilter(bin_img,dims,sgmcolor,sgmspace)    
    PltShowing(blurred,'Tracking...','Blurred', False)



    rows=3
    cols=3
    kernel = np.ones((rows,cols),dtype=np.uint8)
    # Opening = erosion followed by dilation
    erosion = cv2.erode(blurred,kernel,iterations = 1) #thinning, the lines
    dilation = cv2.dilate(blurred,kernel,iterations = 1) #thicking, the lines
    res = np.hstack((erosion,dilation)) #stacking images side-by-side
  
    PltShowing(res,'Tracking...','erosion - dilation', False)



    #######################################
    imask = erosion>0
    # greenInOriframe = np.zeros_like(snip, np.uint8)
    greenInOriframe = np.zeros(src.shape, np.uint8)
    greenInOriframe[imask] = src[imask]

    # 得到彩色的图像
    (b8, g8, r8) = cv2.split(src)
    color_img = cv2.merge([b8 & dilation, g8 & dilation, r8 & dilation])
    # cv2.imshow('color_img', color_img)

    # cv2.waitKey()
    # cv2.destroyAllWindows()
    f = time.time()
    print("Time cost: ",f-t)
    """
    fig = plt.figure('ori-hst-green', figsize=(36, 24), dpi=80, facecolor='w', edgecolor='k')
    fig.canvas.set_window_title('Various RGB Space')
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)
    ax1.title.set_text('Warped Origin')
    ax2.title.set_text('erosion')
    ax3.title.set_text('dilation')
    plt.subplot(1, 3, 1)
    plt.imshow(src, cmap="gray")
    plt.subplot(1, 3, 2)
    plt.imshow(greenInOriframe, cmap="gray")
    plt.subplot(1, 3, 3)
    plt.imshow(color_img)
    plt.show(block=False)
    plt.pause(0.25)
    plt.close()
    # campitch = 45
    # pth = '/media/dom/Elements/data/data_collection/Hirst_wheat/HirstWheatRGBGreen'+ str(campitch) 
    # pth = pth +'.jpg'
    # cv2.imwrite(pth,greenInOriframe)
    """
    
    return greenInOriframe
# for the purpose of handling complex envorinment , such as heavy cannopy and overlapping, etc
def RGBColorSegmtFull(image,Pth_sve4paper,name_pref,pf_start,subpth):
    """
    added more pre-processing operation, this can be focuing on the heavy cannopy and overlapping images
    """    
    # listimages = [image]
    # listtitles = ["Warped"]

    masked = np.copy(image)
    warped = np.copy(image)

    t = time.time()

    # 转换为浮点数进行计算
    fsrc = np.array(image, dtype=np.float32) / 255.0
    (b, g, r) = cv2.split(fsrc)
    green_frame = 2 * g - b - r#由于一张图片最大255，如果我2*g - b -r 超出了255 默认是255，但是如果是小数，怎么也不会超出255,因为uint8类型超过255就是这个数减255截断

    
    
    # listimages.append(green_frame)
    # listtitles.append('green')
    # 求取最大值和最小值
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(green_frame)   
    # 转换为u8类型，进行otsu二值化
    gray_u8 = np.array((green_frame - minVal) / (maxVal - minVal) * 255, dtype=np.uint8)

    im_2gbr= np.copy(gray_u8)

    green_mask = cv2.equalizeHist(gray_u8)
    # listimages.append(green_mask)

    # listtitles.append('equalization')
    # pthname = 'equ'+ name_pref
    # fname = pthname +str(pf_start)+'.jpg'
    # cvSaveImages(green_mask,fname,Pth_sve4paper,False,subpth)
    # greenInOriframe = cv2.bitwise_and(warped, warped, mask=green_mask)
    # listimages.append(greenInOriframe)
    # listtitles.append('equalization')



    # hsv_frame = cv2.cvtColor(masked, cv2.COLOR_BGR2HSV)
    # res = np.hstack((warped,hsv_frame)) #stacking images side-by-side
    # cv2.imwrite('/home/dom/Documents/ARWAC/FarmLanedetection/ref_images/Fig11RGB2HSVrhm.jpg',res)

    ##############################################
    # ori_img_gt = np.copy(warped)#cv2.imread('/media/dom/Elements/data/TX2_Risehm/redblobs3.png')
    # hsv_frame_gt = cv2.cvtColor(ori_img_gt, cv2.COLOR_BGR2HSV)
    # blobCentrods=m_detector.get_cvBlobCentroids(masked,hsv_frame,mask_gt)
    # lanes_gt=get_dic_gt(gt_keys, blobCentrods,bandwith)
    ##############################################    
    # mask = cv2.inRange(hsv_frame, low_green, up_green)
    # hsv_green_mask = cv2.inRange(hsv_frame, lowerLimit, upperLimit)

    #######################################
    """
    # normalizedImg = cv2.normalize(hsv_green_mask,  hsv_green_mask, 0, 255, cv2.NORM_MINMAX)
    # plt.figure('Paper writing')
    # plt.title('Normalized hsv_green_mask')        
    # plt.imshow(normalizedImg,cmap='gray')
    # plt.show(block=True)
    ## slice the green
    imask = mask>0
    greenInOriframe = np.zeros_like(warped, np.uint8)
    greenInOriframe[imask] = warped[imask]
    
    imask = hsv_green_mask>0
    greenInOriframe = np.zeros(green_frame.shape, np.uint8)
    greenInOriframe[imask] = warped[imask]
    listimages.append(greenInOriframe)
    listtitles.append('greenInOriframe')
    
    """
    # Create our shapening kernel, it must equal to one eventually
    kernel_sharpening = np.array([[-1,-1,-1], 
                            [-1, 9,-1],
                            [-1,-1,-1]])

    # applying the sharpening kernel to the input image & displaying it.
    green_mask = cv2.filter2D(green_mask, -1, kernel_sharpening)
    # print('dimension of hsv_green_mask inRanged :', hsv_green_mask.shape)
    green_sharped = cv2.bitwise_and(warped, warped, mask=green_mask)
    # listimages.append(green_mask)
    # listtitles.append('green_roi-sharpening')

    pthname = 'green_sharped'+ name_pref
    fname = pthname +str(pf_start)+'.jpg'
    # cvSaveImages(green_mask,fname,Pth_sve4paper,True,subpth)
    # plt.gcf().suptitle('hsv_green_mask width: ' + str(int(snip.shape[1])) +
                        # '\ncen_hight height: ' + str(int(snip.shape[0])))
    equ = cv2.equalizeHist(green_mask)
    # listimages.append(equ)
    # listtitles.append('equalization')

    # create a CLAHE object (Arguments are optional).
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl1 = clahe.apply(green_mask)
    
    # listimages.append(cl1)
    # listtitles.append('Clahe image')

    #blured image to help with edge detection
    # blurred = cv2.GaussianBlur(hsv_green_mask,(21,21),0); 
    """
    src – Source 8-bit or floating-point, 1-channel or 3-channel image.
    dst – Destination image of the same size and type as src .
    d – Diameter of each pixel neighborhood that is used during filtering. If it is non-positive, it is computed from sigmaSpace .
    sigmaColor – Filter sigma in the color space. A larger value of the parameter means that farther colors within the pixel neighborhood (see sigmaSpace ) will be mixed together, resulting in larger areas of semi-equal color.
    sigmaSpace – Filter sigma in the coordinate space. A larger value of the parameter means that farther pixels will influence each other as long as their colors are close enough (see sigmaColor ). When d>0 , it specifies the neighborhood size regardless of sigmaSpace . Otherwise, d is proportional to sigmaSpace .

    """
    dims=5
    sgmcolor=75
    sgmspace=75
    blurred = cv2.bilateralFilter(green_mask,dims,sgmcolor,sgmspace)
    green_bilateral = cv2.bitwise_and(warped, warped, mask=blurred)
    # listimages.append(blurred)
    # listtitles.append('bilateral Filter')

    # pthname = 'bilaterFlter'+ name_pref
    # fname = pthname +str(pf_start)+'.jpg'
    # cvSaveImages(blurred,fname,Pth_sve4paper,True,subpth)

    # morphological operation kernal - isolate the row - the only purpose
    # small kernel has thining segmentation , which may help with later hough transform, then group the segments of line
    #https://github.com/opencv/opencv/blob/master/samples/python/tutorial_code/imgProc/morph_lines_detection/morph_lines_detection.py
    # 1st erode then dialted , in most case like this :   erode - dilate in above link
    
    rows=3
    cols=3
    kernel = np.ones((rows,cols),dtype=np.uint8)
    # Opening = erosion followed by dilation
    erosion = cv2.erode(blurred,kernel,iterations = 1) #thinning, the lines
    dilation = cv2.dilate(blurred,kernel,iterations = 1) #thicking, the lines
    # res = np.hstack((erosion,dilation)) #stacking images side-by-side
    green_erosion = cv2.bitwise_and(warped, warped, mask=erosion)
    green_dilation = cv2.bitwise_and(warped, warped, mask=dilation)
    # listimages.append(erosion)
    # listtitles.append('erosion')

    # listimages.append(green_dilation)
    # listtitles.append('dilation')

    opening = cv2.morphologyEx(blurred, cv2.MORPH_OPEN, kernel)
    green_opening = cv2.bitwise_and(warped, warped, mask=opening)
    # listimages.append(opening)
    # listtitles.append('opening')

    # cv2.imshow("opening", opening)
    # cv2.waitKey(100)
    # cv2.imwrite('./output/green.png',opening)
    
    # Closing  = Dilation followed by Erosion
    closing = cv2.morphologyEx(blurred, cv2.MORPH_CLOSE, kernel)
    green_closing = cv2.bitwise_and(warped, warped, mask=closing)
    # listimages.append(closing)
    # listtitles.append('closing')
    # plt.gcf().suptitle('closing cols: ' + str(cols) +
                        # '\nsigma rows: ' + str(rows))


    # 转换为浮点数进行计算
    # fsrc = np.array(image, dtype=np.float32) / 255.0
    # (b, g, r) = cv2.split(fsrc)
    # green_frame = 2 * g - b - r#由于一张图片最大255，如果我2*g - b -r 超出了255 默认是255，但是如果是小数，怎么也不会超出255,因为uint8类型超过255就是这个数减255截断
    
    thrimage = erosion#cdilation#erosion#closing#opening
    # pthname = 'ersion'+ name_pref
    # fname = pthname +str(pf_start)+'.jpg'
    # cvSaveImages(thrimage,fname,Pth_sve4paper,True,subpth)

    # 求取最大值和最小值
    # (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(thrimage)    
    # 转换为u8类型，进行otsu二值化
    # gray_u8 = np.array((gray - minVal) / (maxVal - minVal) * 255, dtype=np.uint8)
    # gray_u8 = cv2.equalizeHist(gray_u8)
    # ret,th1 = cv2.threshold(gray_u8, -1.0, 255, cv2.THRESH_OTSU)
    # cv2.imshow('bin_img', bin_img)
    ######################################


    

    blcksiz = 3
    bordsize = 2
    maxval =225
    ret,th1 = cv2.threshold(thrimage,-1.0,maxval,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    green_OTSU = cv2.bitwise_and(warped, warped, mask=th1)
    # listimages.append(green_OTSU)
    # listtitles.append('adapt OTSU-th1')

    # pthname = 'otsu'+ name_pref
    # fname = pthname +str(pf_start)+'.jpg'
    # cvSaveImages(th1,fname,Pth_sve4paper,True,subpth)


    th2 = cv2.adaptiveThreshold(thrimage,maxval,cv2.ADAPTIVE_THRESH_MEAN_C,\
        cv2.THRESH_BINARY,blcksiz,bordsize)
    green_MEAN = cv2.bitwise_and(warped, warped, mask=th2)
    # listimages.append(th2)
    # listtitles.append('adapt MEANC-th2')
    # plt.gcf().suptitle('MEAN_C dim: ' + str(blcksiz) +
                        # '\nbordsize: ' + str(bordsize))
    # pthname = 'MEANC'+ name_pref
    # fname = pthname +str(pf_start)+'.jpg'
    # cvSaveImages(th2,fname,Pth_sve4paper,True,subpth)

    th3 = cv2.adaptiveThreshold(thrimage,maxval,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
        cv2.THRESH_BINARY,blcksiz,bordsize)

    green_GAUSSIAN = cv2.bitwise_and(warped, warped, mask=th3)
    # listimages.append(th3)
    # listtitles.append('adapt GAUSSC-th3')
    # plt.gcf().suptitle('GAUSSIAN_C dim: ' + str(blcksiz) +
                        # '\nbordsize: ' + str(bordsize))
    # pthname = 'GAUSSIAN'+ name_pref
    # fname = pthname +str(pf_start)+'.jpg'
    # cvSaveImages(th3,fname,Pth_sve4paper,True,subpth)

    # convolute with proper kernels
    ddepth = cv2.CV_16S#cv2.CV_64F#cv2.CV_16S
    kernel_size =3

    sobelinput =th1#, th2, th3
    """
    laplacian = cv2.Laplacian(th3,ddepth,kernel_size)
    laplacian = cv2.convertScaleAbs(laplacian)
    listimages.append(laplacian)
    listtitles.append('laplacian')
    """
    sobelx = cv2.Sobel(sobelinput,ddepth,1,0,ksize=kernel_size)  # x
    sobelx = cv2.convertScaleAbs(sobelx)
    green_sobelx = cv2.bitwise_and(warped, warped, mask=sobelx)
    # listimages.append(green_sobelx)
    # listtitles.append('sobelx')
    
    # pthname = 'sobelx'+ name_pref
    # fname = pthname +str(pf_start)+'.jpg'
    # cvSaveImages(sobelx,fname,Pth_sve4paper,True,subpth)


    sobely = cv2.Sobel(sobelinput,ddepth,0,1,ksize=kernel_size)  # y
    sobely = cv2.convertScaleAbs(sobely)
    green_sobely = cv2.bitwise_and(warped, warped, mask=sobely)
    # listimages.append(green_sobely)
    # listtitles.append('sobely')
    
    # pthname = 'sobely'+ name_pref
    # fname = pthname +str(pf_start)+'.jpg'
    # cvSaveImages(sobely,fname,Pth_sve4paper,True,subpth)
    # Plot the all resulted processed images
    ncols = 5
    nrows = 5
    

    # plot first image on its own row

    """fig = plt.figure('green ROI', figsize=(36, 24), dpi=80, facecolor='w', edgecolor='k')
    # fig.canvas.set_window_title('Various RGB Space')  
    plt.subplot(nrows,ncols,1)
    plt.imshow(listimages[0],'gray')
    plt.title(listtitles[0])
    plt.xticks([]),plt.yticks([])
    
    numofl = len(listimages)
    print('total number of displays: ',numofl)
    for i in range(1,ncols*(nrows-1)+1):#range(0,ncols*(nrows-1)+1): #range(0,ncols*(nrows-1)+1): if separated one row above 
        try:
            # plt.subplot(nrows,ncols,i+ncols),plt.imshow(listimages[i],'gray')
            plt.subplot(nrows,ncols,i+ncols)
            plt.imshow( cv2.cvtColor(listimages[i],cv2.COLOR_BGR2RGB),'gray')
            plt.title(listtitles[i])
            plt.xticks([]),plt.yticks([])
        except IndexError:
            print ('IndexError in plot.')
            break

    #plt.gcf().suptitle(filename)
    plt.gcf().canvas.set_window_title(name_pref)
    plt.show(block=True)
    plt.pause(0.05)
    plt.close()"""
    
    green_wheat = np.copy(green_OTSU)
    """
    using sobel the features are extracted only from the edges, which is not this case , as we need central features, 
    but previous method are extract from edge feature, and clustering later  for hough transofrm
    # green_wheat = np.copy(green_sobelx)
    # green_wheat = np.copy(green_sobely)
    """
    return im_2gbr,green_wheat



def HSVColorSegmtFull(image,lowerLimit,upperLimit,name_pref):
    """
    added more pre-processing operation, this can be focuing on the heavy cannopy and overlapping images
    """    
    listimages = [image]
    listtitles = ["Warped"]

    masked = np.copy(image)
    warped = np.copy(image)

    hsv_frame = cv2.cvtColor(masked, cv2.COLOR_BGR2HSV)
    res = np.hstack((warped,hsv_frame)) #stacking images side-by-side
    # cv2.imwrite('/home/dom/Documents/ARWAC/FarmLanedetection/ref_images/Fig11RGB2HSVrhm.jpg',res)
    listimages.append(hsv_frame)
    listtitles.append('hsv_frame')
    ##############################################
    # ori_img_gt = np.copy(warped)#cv2.imread('/media/dom/Elements/data/TX2_Risehm/redblobs3.png')
    # hsv_frame_gt = cv2.cvtColor(ori_img_gt, cv2.COLOR_BGR2HSV)
    # blobCentrods=m_detector.get_cvBlobCentroids(masked,hsv_frame,mask_gt)
    # lanes_gt=get_dic_gt(gt_keys, blobCentrods,bandwith)
    ##############################################    
    # mask = cv2.inRange(hsv_frame, low_green, up_green)
    hsv_green_mask = cv2.inRange(hsv_frame, lowerLimit, upperLimit)

    #######################################
    """
    # normalizedImg = cv2.normalize(hsv_green_mask,  hsv_green_mask, 0, 255, cv2.NORM_MINMAX)
    # plt.figure('Paper writing')
    # plt.title('Normalized hsv_green_mask')        
    # plt.imshow(normalizedImg,cmap='gray')
    # plt.show(block=True)
    ## slice the green
    imask = mask>0
    greenInOriframe = np.zeros_like(warped, np.uint8)
    greenInOriframe[imask] = warped[imask]
    greenInOriframe = cv2.bitwise_and(snip, snip, mask=normalizedImg)
    """
    imask = hsv_green_mask>0
    greenInOriframe = np.zeros(hsv_frame.shape, np.uint8)
    greenInOriframe[imask] = warped[imask]
    listimages.append(greenInOriframe)
    listtitles.append('greenInOriframe')
    # print ("hsv_green_mask and greenInOriframe dimension:", hsv_green_mask.shape, greenInOriframe.shape)
    # stacked_hsv_3d = np.stack((hsv_green_mask,)*3, axis=-1)
    # res = np.hstack((stacked_hsv_3d,greenInOriframe)) #stacking images side-by-side
    # plt.figure('Paper writing',figsize=(36,24))
    # plt.title('Green space HSV and RGB image')        
    # plt.imshow(res,cmap='gray')
    # plt.show(block=True)
    # plt.pause(0.25)
    # plt.close()
    # cv2.imwrite('/home/dom/Documents/ARWAC/FarmLanedetection/ref_images/Fig12GREEHSVandRGBrhm.jpg',res)
    #########################################
    # Create our shapening kernel, it must equal to one eventually
    kernel_sharpening = np.array([[-1,-1,-1], 
                            [-1, 9,-1],
                            [-1,-1,-1]])

    # applying the sharpening kernel to the input image & displaying it.
    hsv_green_mask = cv2.filter2D(hsv_green_mask, -1, kernel_sharpening)
    # print('dimension of hsv_green_mask inRanged :', hsv_green_mask.shape)
    listimages.append(hsv_green_mask)
    listtitles.append('hsv_green_mask-sharpened')
    # plt.gcf().suptitle('hsv_green_mask width: ' + str(int(snip.shape[1])) +
                        # '\ncen_hight height: ' + str(int(snip.shape[0])))

    """
    hist,bins = np.histogram(hsv_green_mask.flatten(),256,[0,256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * float(hist.max()) / cdf.max()
    plt.title('historgram')
    plt.plot(cdf_normalized, color = 'b')
    plt.hist(hsv_green_mask.flatten(),256,[0,256], color = 'r')
    plt.xlim([0,256])
    plt.legend(('cdf','histogram'), loc = 'upper left')
    plt.show(block=False)
    plt.pause(0.25)
    plt.close()
    # plt.show()  

    cdf_m = np.ma.masked_equal(cdf,0)
    cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
    cdf = np.ma.filled(cdf_m,0).astype('uint8')
    img2 = cdf[hsv_green_mask]
    hist,bins = np.histogram(img2.flatten(),256,[0,256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * float(hist.max()) / cdf.max()
    plt.title('historgram normalized')
    plt.plot(cdf_normalized, color = 'b')
    plt.hist(img2.flatten(),256,[0,256], color = 'r')
    plt.xlim([0,256])
    plt.legend(('cdf','histogram'), loc = 'upper left')
    plt.show(block=False)
    plt.pause(0.5)
    plt.close()
    """

    equ = cv2.equalizeHist(hsv_green_mask)
    # res = np.hstack((hsv_green_mask,equ)) #stacking images side-by-side
    # cv.imwrite('res.png',res)
    # plt.title('before and after image')
    # plt.imshow(res,cmap='gray')
    # plt.show(block=True)
    # plt.show() 
    listimages.append(equ)
    listtitles.append('His equilization')

    # create a CLAHE object (Arguments are optional).
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl1 = clahe.apply(hsv_green_mask)
    # rescl1 = np.hstack((equ,cl1))
    # plt.title('His and Clahe image')
    # plt.imshow(rescl1,cmap='gray')
    # plt.show(block=True)
    # plt.show()
    listimages.append(cl1)
    listtitles.append('Clahe image')

    #blured image to help with edge detection
    # blurred = cv2.GaussianBlur(hsv_green_mask,(21,21),0); 
    """
    src – Source 8-bit or floating-point, 1-channel or 3-channel image.
    dst – Destination image of the same size and type as src .
    d – Diameter of each pixel neighborhood that is used during filtering. If it is non-positive, it is computed from sigmaSpace .
    sigmaColor – Filter sigma in the color space. A larger value of the parameter means that farther colors within the pixel neighborhood (see sigmaSpace ) will be mixed together, resulting in larger areas of semi-equal color.
    sigmaSpace – Filter sigma in the coordinate space. A larger value of the parameter means that farther pixels will influence each other as long as their colors are close enough (see sigmaColor ). When d>0 , it specifies the neighborhood size regardless of sigmaSpace . Otherwise, d is proportional to sigmaSpace .

    """
    hsv_green_mask=cl1
    # hsv_green_mask=cl1
    dims=5
    sgmcolor=75
    sgmspace=75
    blurred = cv2.bilateralFilter(hsv_green_mask,dims,sgmcolor,sgmspace)
    green_bilateral = cv2.bitwise_and(warped, warped, mask=blurred)
    listimages.append(green_bilateral)
    listtitles.append('Bila Filter')
    # plt.gcf().suptitle('filtered dim: ' + str(dims) +
                        # '\nsigma color: ' + str(sgmcolor)+
                        # '\nsigma space: ' + str(sgmspace))

    # cv2.imshow("blurred", blurred)
    # cv2.waitKey(100)
    # cv2.imwrite('./output_images/blurred_wheat.png',blurred) 

    # morphological operation kernal - isolate the row - the only purpose
    # small kernel has thining segmentation , which may help with later hough transform, then group the segments of line
    #https://github.com/opencv/opencv/blob/master/samples/python/tutorial_code/imgProc/morph_lines_detection/morph_lines_detection.py
    # 1st erode then dialted , in most case like this :   erode - dilate in above link
    
    rows=5
    cols=5
    kernel = np.ones((rows,cols),dtype=np.uint8)
    # Opening = erosion followed by dilation
    erosion = cv2.erode(blurred,kernel,iterations = 1) #thinning, the lines
    dilation = cv2.dilate(blurred,kernel,iterations = 1) #thicking, the lines
    # res = np.hstack((erosion,dilation)) #stacking images side-by-side
    green_erosion = cv2.bitwise_and(warped, warped, mask=erosion)
    green_dilation = cv2.bitwise_and(warped, warped, mask=dilation)
    listimages.append(green_erosion)
    listtitles.append('erosion')

    listimages.append(green_dilation)
    listtitles.append('dilation')

    opening = cv2.morphologyEx(blurred, cv2.MORPH_OPEN, kernel)
    green_opening = cv2.bitwise_and(warped, warped, mask=opening)
    listimages.append(green_opening)
    listtitles.append('opening')

    # cv2.imshow("opening", opening)
    # cv2.waitKey(100)
    # cv2.imwrite('./output/green.png',opening)
    
    # Closing  = Dilation followed by Erosion
    closing = cv2.morphologyEx(blurred, cv2.MORPH_CLOSE, kernel)
    green_closing = cv2.bitwise_and(warped, warped, mask=closing)
    listimages.append(green_closing)
    listtitles.append('closing')
    # plt.gcf().suptitle('closing cols: ' + str(cols) +
                        # '\nsigma rows: ' + str(rows))

    thrimage = erosion#cdilation#erosion#closing#opening
    blcksiz = 5
    bordsize = 2
    maxval =225
    ret,th1 = cv2.threshold(thrimage,0,maxval,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    green_OTSU = cv2.bitwise_and(warped, warped, mask=th1)
    listimages.append(green_OTSU)
    listtitles.append('adapt OTSU-th1')


    th2 = cv2.adaptiveThreshold(thrimage,maxval,cv2.ADAPTIVE_THRESH_MEAN_C,\
        cv2.THRESH_BINARY,blcksiz,bordsize)
    green_MEAN = cv2.bitwise_and(warped, warped, mask=th2)
    listimages.append(green_MEAN)
    listtitles.append('adapt MEANC-th2')
    # plt.gcf().suptitle('MEAN_C dim: ' + str(blcksiz) +
                        # '\nbordsize: ' + str(bordsize))

    th3 = cv2.adaptiveThreshold(thrimage,maxval,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
        cv2.THRESH_BINARY,blcksiz,bordsize)

    green_GAUSSIAN = cv2.bitwise_and(warped, warped, mask=th3)
    listimages.append(green_GAUSSIAN)
    listtitles.append('adapt GAUSSC-th3')
    # plt.gcf().suptitle('GAUSSIAN_C dim: ' + str(blcksiz) +
                        # '\nbordsize: ' + str(bordsize))

    # convolute with proper kernels
    ddepth = cv2.CV_16S#cv2.CV_64F#cv2.CV_16S
    kernel_size =9

    sobelinput = th1#, th2, th3
    """
    laplacian = cv2.Laplacian(th3,ddepth,kernel_size)
    laplacian = cv2.convertScaleAbs(laplacian)
    listimages.append(laplacian)
    listtitles.append('laplacian')
    """
    sobelx = cv2.Sobel(sobelinput,ddepth,1,0,ksize=kernel_size)  # x
    sobelx = cv2.convertScaleAbs(sobelx)
    green_sobelx = cv2.bitwise_and(warped, warped, mask=sobelx)
    listimages.append(green_sobelx)
    listtitles.append('sobelx')



    sobely = cv2.Sobel(sobelinput,ddepth,0,1,ksize=kernel_size)  # y
    sobely = cv2.convertScaleAbs(sobely)
    green_sobely = cv2.bitwise_and(warped, warped, mask=sobely)
    listimages.append(green_sobely)
    listtitles.append('sobely')
    # Plot the all resulted processed images
    ncols = 5
    nrows = 5
    """
    # plot first image on its own row

    fig = plt.figure('green ROI', figsize=(36, 24), dpi=80, facecolor='w', edgecolor='k')
    # fig.canvas.set_window_title('Various RGB Space')  
    plt.subplot(nrows,ncols,1)
    plt.imshow(listimages[0],'gray')
    plt.title(listtitles[0])
    plt.xticks([]),plt.yticks([])
    
    numofl = len(listimages)
    print('total number of displays: ',numofl)
    for i in range(1,ncols*(nrows-1)+1):#range(0,ncols*(nrows-1)+1): #range(0,ncols*(nrows-1)+1): if separated one row above 
        try:
            # plt.subplot(nrows,ncols,i+ncols),plt.imshow(listimages[i],'gray')
            plt.subplot(nrows,ncols,i+ncols)
            plt.imshow(cv2.cvtColor( listimages[i], cv2.COLOR_BGR2RGB),cmap='gray')
            plt.title(listtitles[i])
            plt.xticks([]),plt.yticks([])
        except IndexError:
            print ('IndexError in plot.')
            break

    #plt.gcf().suptitle(filename)
    plt.gcf().canvas.set_window_title(name_pref)
    plt.show(block=False)
    plt.pause(0.25)
    plt.close()
    """
    green_wheat = np.copy(green_sobely)

    return green_wheat
        # blurred = green_sobelx#laplacian#th3#th2#sobelx#th3
        # erosion = blurred
        # erosion = cv2.erode(blurred,kernel,iterations = 1) #thinning, the lines
        # listimages.append(blurred)
        # listtitles.append('blurred')

        # plt.figure('Tracking...')
        # plt.title('blurred')
        # plt.imshow(blurred,cmap='gray')
        # plt.show(block=False)
        # plt.pause(0.25)
        # plt.close()

        # res = np.hstack((sobelx,sobely)) #stacking images side-by-side
        # cv2.imwrite('/home/dom/ARWAC/Documents/ARWAC/domReport/ref_images/Fig16sobelx.jpg',sobelx)
        # cv2.imwrite('/home/dom/ARWAC/Documents/ARWAC/domReport/ref_images/Fig16sobely.jpg',sobely)
        # cv2.imwrite('/home/dom/ARWAC/Documents/ARWAC/domReport/ref_images/Fig16sobelxy.jpg',res)
        # plt.figure('Tracking...')
        # plt.title('sobelx-y')
        # plt.imshow(res,cmap='gray')
        # plt.show(block=False)
        # plt.pause(0.25)
        # plt.close()
        
        # minthres=50#50
        # maxtres=200#200

        # sobelxedges = cv2.Canny(sobelx, minthres, maxtres) #150
        # sobelyedges = cv2.Canny(sobely, minthres, maxtres) #150
        # res = np.hstack((sobelxedges,sobelyedges)) #stacking images side-by-side




        # green_wheat = cv2.bitwise_and(warped, warped, mask=blurred)
        # listimages.append(blurred)
        # listtitles.append('input to Canny')
        # edges = cv2.Canny(blurred, minthres, maxtres) #150
        # plt.figure('Tracking...')
        # plt.title('edges')
        # plt.imshow(edges,cmap='gray')
        # plt.show(block=False)
        # plt.pause(0.25)
        # plt.close()
        # plt.show(block=False)
        # plt.pause(1)
        # plt.close()
        # cv2.imshow("edges", edges)
        # cv2.waitKey(100)
        # listimages.append(edges)
        # listtitles.append('Canny')
        # plt.gcf().suptitle('Canny minthres: ' + str(minthres) +
                            # '\nmaxtres: ' + str(maxtres))
        # cv2.imwrite('./output_images/edges_wheat.png',edges)
        # cv2.imwrite('/home/dom/Documents/ARWAC/FarmLanedetection/ref_images/Fig22LineEdgesrhm.jpg',edges)
        #######################################


def HSVColorSegmt(image,lowerLimit,upperLimit):
    
    # image = cv2.imread(r'Image.jpg') #load your image
    hsv_frame = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) #convert the image into hsv

   
    """
    green = np.uint8([[[0, 255, 0]]])  #green color
    hsvGreen = cv2.cvtColor(green, cv2.COLOR_BGR2HSV) #hsv value of green color 
    print('hsvGreen = ',hsvGreen) 

    lowerLimit = hsvGreen[0][0][0] - rng, 25, 25  # range of green color lower limit and upper limit
    upperLimit = hsvGreen[0][0][0] + rng, 255, 255
    
    print('lowerLimit = ', lowerLimit)
    print('upperLimit = ', upperLimit)
    
    # red = np.uint8([[[0, 0, 255]]]) #red color
    # hsvred = cv2.cvtColor(red, cv2.COLOR_BGR2HSV) #hsv value of red color
    # print(hsvred)
    # lower = hsvred[0][0][0] - 10, 100, 100 # range of red color lower limit and upper limit
    # upper = hsvred[0][0][0] + 10, 255, 255
    # print(upper)
    # print(lower)
    
    # image = cv2.imread(r'Image.jpg') #load your image
    # hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) #convert the image into hsv

    lowerLimit = [20,20,20]
    upperLimit = [100,200,200]

    cv2.imwrite('/media/dom/Elements/data/data_collection/Hirst_wheat/Fig11WheatRGB2HSV.jpg',res)
    #########################################
    H,S,V = cv2.split(hsv_frame)
    S_equalized= cv2.equalizeHist(S)
    V_equalized= cv2.equalizeHist(V)
    hsv_adjust = cv2.merge((H,S,V))

    v = hsv_frame.T [2] .flatten (). mean ()
    print ("Value:% .2f"% (v))
    # Average of H values ​​where H is less than 220
    mean = H [H<220].mean ()
    print (mean) # 66.82431244444444
    # Images are always displayed as they are BGR. If you want to show your HSV image, you need to convert to BGR first:
    out = cv2.cvtColor(hsv_adjust, cv2.COLOR_HSV2BGR)        
    fig = plt.figure('ori-hsv-adjust', figsize=(36, 24), dpi=80, facecolor='w', edgecolor='k')
    fig.canvas.set_window_title('Various HSV Space')
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)
    ax1.title.set_text('Warped to Origin')
    ax2.title.set_text('hsv_frame')
    ax3.title.set_text('hsv adjusted')
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(snip, cv2.COLOR_BGR2RGB) , cmap="gray")
    plt.subplot(1, 3, 2)
    plt.imshow(cv2.cvtColor(hsv_frame, cv2.COLOR_BGR2RGB) , cmap="gray")
    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(out, cv2.COLOR_BGR2RGB) )
    plt.show(block=True)
    plt.close()

    """
    lg = np.array(lowerLimit) #range of green color
    ug = np.array(upperLimit)
    hsv_green_mask = cv2.inRange(hsv_frame, lg, ug) #green masked image
    # cv2.imshow('green', hsv_green_mask) #show the image 

    # lr = np.array(lower) #range of red color
    # ur = np.array(upper)

    # red_mask = cv2.inRange(hsv, lr, ur) #red masked image
    # cv2.imshow('red', red_mask)  #show the image 
    normalizedImg = cv2.normalize(hsv_green_mask,  hsv_green_mask, 0, 255, cv2.NORM_MINMAX)
    normalizedImg = cv2.equalizeHist(normalizedImg)
    (thresh, bin_img) = cv2.threshold(normalizedImg, -1.0, 255, cv2.THRESH_OTSU)
    greenInOriframe = cv2.bitwise_and(image, image, mask=bin_img)

    """
    print ("green_mask and greenInOriframe dimension:", green_mask.shape, greenInOriframe.shape)
    stacked_hsv_3d = np.stack((green_mask,)*3, axis=-1)
    res = np.hstack((stacked_hsv_3d,greenInOriframe)) #stacking images side-by-side
    plt.figure('Paper writing',figsize=(36,24))
    plt.title('Green space HSV and RGB image')        
    plt.imshow(cv2.cvtColor(res, cv2.COLOR_BGR2RGB),cmap='gray')
    plt.show(block=True)
    plt.pause(0.5)
    plt.close()
    cv2.imwrite('/media/dom/Elements/data/data_collection/Hirst_wheat/Fig12WheatGREEHSVandRGB.jpg',res)
     
    """

    # slice the green
    # imask = normalizedImg>0
    # greenInOriframe = np.zeros_like(snip, np.uint8)
    # greenInOriframe = np.zeros(image.shape, np.uint8)
    # greenInOriframe[imask] = image[imask]


    fig = plt.figure('ori-hsv-green', figsize=(36, 24), dpi=80, facecolor='w', edgecolor='k')
    fig.canvas.set_window_title('Various HSV Space')
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)
    ax1.title.set_text('Warped to Origin')
    ax2.title.set_text('normalizedImg hsv')
    ax3.title.set_text('green after HSV Otsu')
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB) , cmap="gray")
    plt.subplot(1, 3, 2)
    plt.imshow(cv2.cvtColor(normalizedImg, cv2.COLOR_BGR2RGB) , cmap="gray")
    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(greenInOriframe, cv2.COLOR_BGR2RGB) )
    plt.show(block=True)
    plt.close()

    # return greenInOriframe
    ####################################################
    """
    src – Source 8-bit or floating-point, 1-channel or 3-channel image.
    dst – Destination image of the same size and type as src .
    d – Diameter of each pixel neighborhood that is used during filtering. If it is non-positive, it is computed from sigmaSpace .
    sigmaColor – Filter sigma in the color space. A larger value of the parameter means that farther colors within the pixel neighborhood (see sigmaSpace ) will be mixed together, resulting in larger areas of semi-equal color.
    sigmaSpace – Filter sigma in the coordinate space. A larger value of the parameter means that farther pixels will influence each other as long as their colors are close enough (see sigmaColor ). When d>0 , it specifies the neighborhood size regardless of sigmaSpace . Otherwise, d is proportional to sigmaSpace .
    #######################################################
    """
        
    dims=5
    sgmcolor=75
    sgmspace=75
    blurred = cv2.bilateralFilter(bin_img,dims,sgmcolor,sgmspace)
    green_bilateral = cv2.bitwise_and(image, image, mask=blurred)

    plt.figure('Tracking...')
    plt.title('bilateral')
    plt.imshow(blurred,cmap='gray')
    plt.show(block=False)
    plt.pause(0.5)
    plt.close()
    # plt.show(block=False)
    # plt.pause(1)
    # plt.close()

    # cv2.imshow("blurred", blurred)
    # cv2.waitKey(100)
    cv2.imwrite('/media/dom/Elements/data/data_collection/Hirst_wheat/Fig12WheatBilateral.png',blurred) 

    # morphological operation kernal - isolate the row - the only purpose
    # small kernel has thining segmentation , which may help with later hough transform, then group the segments of line
    #https://github.com/opencv/opencv/blob/master/samples/python/tutorial_code/imgProc/morph_lines_detection/morph_lines_detection.py
    # 1st erode then dialted , in most case like this :   erode - dilate in above link
    rows=5
    cols=5
    kernel = np.ones((rows,cols),dtype=np.uint8)
    # Opening = erosion followed by dilation
    erosion = cv2.erode(blurred,kernel,iterations = 1) #thinning, the lines
    dilation = cv2.dilate(blurred,kernel,iterations = 1) #thicking, the lines
    res = np.hstack((erosion,dilation)) #stacking images side-by-side

        
        
    plt.figure('Tracking...')
    plt.title('erosion - dilation')
    plt.imshow(res,cmap='gray')
    plt.show(block=False)
    plt.pause(0.5)
    plt.close()
 

    green_erosion = cv2.bitwise_and(image, image, mask=erosion)
    green_dilation = cv2.bitwise_and(image, image, mask=dilation)    

      
    #######################################################
    # Opening = erosion followed by dilation
    opening = cv2.morphologyEx(blurred, cv2.MORPH_OPEN, kernel)
    green_opening = cv2.bitwise_and(image, image, mask=opening)
    
       
    # Closing  = Dilation followed by Erosion
    closing = cv2.morphologyEx(blurred, cv2.MORPH_CLOSE, kernel)
    green_closing = cv2.bitwise_and(image, image, mask=closing)
    plt.figure('Tracking...')
    plt.title('closing')
    plt.imshow(blurred,cmap='gray')   
    # plt.show(block=False)
    plt.show(block=False)
    plt.pause(0.5)
    plt.close()

    # cv2.imshow("closing", closing)
    # cv2.waitKey(100)
    # cv2.imwrite('./output/green.png',opening) 

    fig = plt.figure('bilateralFilter-errosion-dilation', figsize=(36, 24), dpi=80, facecolor='w', edgecolor='k')
    fig.canvas.set_window_title('various operation')

    ax = fig.add_subplot(221)    
    plt.subplot(2, 2, 1)
    plt.imshow(cv2.cvtColor(green_bilateral, cv2.COLOR_BGR2RGB) , cmap="gray")
    ax.title.set_text('bilateralFilter')

    ax = fig.add_subplot(222)    
    plt.subplot(2, 2, 2)
    plt.imshow(cv2.cvtColor(green_erosion, cv2.COLOR_BGR2RGB) , cmap="gray")
    ax.title.set_text('errosion')

    ax = fig.add_subplot(223)    
    plt.subplot(2, 2, 3)
    plt.imshow(cv2.cvtColor(green_dilation, cv2.COLOR_BGR2RGB) )
    ax.title.set_text('dilation')

    ax = fig.add_subplot(224) 
    plt.subplot(2, 2, 4)
    plt.imshow(cv2.cvtColor(green_opening, cv2.COLOR_BGR2RGB) )
    ax.title.set_text('opening')
 
    plt.show(block=True)
    plt.close()


    edges = cv2.Canny(blurred, 50, 200) #150
    res = np.hstack((blurred,edges)) #stacking images side-by-side
    plt.figure('Tracking...',figsize=(36,24))
    plt.title('input - edges')
    plt.imshow(res,cmap='gray')
    # plt.show()
    plt.show(block=False)
    plt.pause(1)
    plt.close()
    # cv2.imshow("edges", edges)
    # cv2.waitKey(100)
    # cv2.imwrite('/media/dom/Elements/data/data_collection/Hirst_wheat/edges.png',edges)
    # cv2.imwrite('/media/dom/Elements/data/data_collection/Hirst_wheat/Fig22WheatLineEdgesCh.jpg',edges)
    ####################################################    
    
    greenInOriframe = np.copy(green_erosion)
    campitch=45
    pth = '/media/dom/Elements/data/data_collection/Hirst_wheat/HirstWheatHSVGreenROI'+ str(campitch)
    pth = pth +'.jpg'
    cv2.imwrite(pth,greenInOriframe)

    return greenInOriframe



def BuildHSVColorModel(image,rng):
    
    # image = cv2.imread(r'Image.jpg') #load your image
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) #convert the image into hsv
    green = np.uint8([[[0, 255, 0]]])  #green color
    hsvGreen = cv2.cvtColor(green, cv2.COLOR_BGR2HSV) #hsv value of green color 
    print('hsvGreen = ',hsvGreen) 

    lowerLimit = hsvGreen[0][0][0] - rng, 25, 25  # range of green color lower limit and upper limit
    upperLimit = hsvGreen[0][0][0] + rng, 255, 255
    
    print('lowerLimit = ', lowerLimit)
    print('upperLimit = ', upperLimit)


    # red = np.uint8([[[0, 0, 255]]]) #red color
    # hsvred = cv2.cvtColor(red, cv2.COLOR_BGR2HSV) #hsv value of red color
    # print(hsvred)

    # lower = hsvred[0][0][0] - 10, 100, 100 # range of red color lower limit and upper limit
    # upper = hsvred[0][0][0] + 10, 255, 255

    # print(upper)
    # print(lower)

    # image = cv2.imread(r'Image.jpg') #load your image
    # hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) #convert the image into hsv
    lowerLimit = [20,20,20]
    upperLimit = [100,200,200]
    lg = np.array(lowerLimit) #range of green color
    ug = np.array(upperLimit)
    green_mask = cv2.inRange(hsv, lg, ug) #green masked image
    # cv2.imshow('green', green_mask) #show the image 

    # lr = np.array(lower) #range of red color
    # ur = np.array(upper)

    # red_mask = cv2.inRange(hsv, lr, ur) #red masked image
    # cv2.imshow('red', red_mask)  #show the image 
    normalizedImg = cv2.normalize(green_mask,  green_mask, 0, 255, cv2.NORM_MINMAX)
    greenInOriframe = cv2.bitwise_and(image, image, mask=normalizedImg)

    # imask = erosion>0
    # # greenInOriframe = np.zeros_like(snip, np.uint8)
    # greenInOriframe = np.zeros(src.shape, np.uint8)
    # greenInOriframe[imask] = src[imask]

    fig = plt.figure('ori-hsv-adjust', figsize=(36, 24), dpi=80, facecolor='w', edgecolor='k')
    fig.canvas.set_window_title('Various HSV Space')
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)
    ax1.title.set_text('Warped to Origin')
    ax2.title.set_text('normalizedImg hsv')
    ax3.title.set_text('greenInOriframe')
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB) , cmap="gray")
    plt.subplot(1, 3, 2)
    plt.imshow(cv2.cvtColor(normalizedImg, cv2.COLOR_BGR2RGB) , cmap="gray")
    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(greenInOriframe, cv2.COLOR_BGR2RGB) )
    plt.show(block=True)
    plt.close()
    
    campitch=45
    pth = '/media/dom/Elements/data/data_collection/Hirst_wheat/HirstWheatHSVGreen'+ str(campitch)
    pth = pth +'.jpg'
    cv2.imwrite(pth,greenInOriframe)

    return greenInOriframe
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
        