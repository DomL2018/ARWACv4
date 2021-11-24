import cv2
import numpy as np
# from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
# plt.switch_backend('agg')

import matplotlib
# matplotlib.use('TkAgg')
from skimage.morphology import skeletonize
from skimage.filters import threshold_otsu
from skimage.morphology import medial_axis

# http://opencvpython.blogspot.com/2012/05/skeletonization-using-opencv-python.html
def cv_ToplogicalSkel(gscale):
    # Skeletonization is the process of thinning the regions of interest to their binary constituents. This makes it easier for to perform pattern recognition.
    size = np.size(gscale) #returns the product of the array dimensions
    skel = np.zeros(gscale.shape,np.uint8) #array of zeros
    ret,gscale = cv2.threshold(gscale,128,255,0) #thresholding the image
    element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
    done = False
    while( not done):
        eroded = cv2.erode(gscale,element)
        temp = cv2.dilate(eroded,element)
        temp = cv2.subtract(gscale,temp)
        skel = cv2.bitwise_or(skel,temp)
        gscale = eroded.copy()
        zeros = size - cv2.countNonZero(gscale)
        plt.figure('skl literation')
        plt.imshow(gscale)
        mng = plt.get_current_fig_manager()
        mng.frame.Maximize(True)
        plt.show(block=False)
        plt.pause(0.5)
        plt.close()
        if zeros==size:
            done = True

    return gscale
# https://scikit-image.org/docs/dev/auto_examples/edges/plot_skeleton.html
def skim_ToplogicalSkel(gscale,fullpth,threshold=64):

    threshold = threshold_otsu(gscale) #input value
    blobs = (gscale > threshold) #binary is when less than trheshold

   

    skeleton_cv = skeletonize(blobs)
    skeleton_lee = skeletonize(blobs, method='lee')

    # media distance tranform, the skeleton will stay in the middle of segemetation
    ret1,th1 = cv2.threshold(gscale,64,255,cv2.THRESH_BINARY)
    skel, distance = medial_axis(th1, return_distance=True)
    # Distance to the background for pixels of the skeleton
    mask_ske = distance * skel

    fig, axes = plt.subplots(2, 2, figsize=(8, 4), sharex=True, sharey=True)
    ax = axes.ravel()

    ax[0].imshow(blobs, cmap=plt.cm.gray)
    ax[0].set_title('original')
    ax[0].axis('off')

    ax[1].imshow(skeleton_cv, cmap=plt.cm.gray)
    ax[1].set_title('skeletonize')
    ax[1].axis('off')

    ax[2].imshow(skeleton_lee, cmap=plt.cm.gray)
    ax[2].set_title('skeletonize (Lee 94)')
    ax[2].axis('off')

    ax[3].imshow(mask_ske, cmap=plt.cm.gray)
    ax[3].set_title('distance transform')
    ax[3].axis('off')
 

    fig.tight_layout()
    # fig.savefig(fullpth)
    # fig.savefig(fullpth, dpi=None, facecolor='w', edgecolor='w',orientation='portrait', papertype=None, format=None,transparent=False, bbox_inches=None, pad_inches=0.1, frameon=None, metadata=None)
    plt.show(block=True)
    plt.pause(0.05)
    plt.close()


    return skeleton_cv, skeleton_lee, mask_ske




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