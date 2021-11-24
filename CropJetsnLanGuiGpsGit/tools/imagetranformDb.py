from __future__ import print_function
import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
# from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
# plt.switch_backend('agg')
from skimage.morphology import skeletonize
from skimage.filters import threshold_otsu
from skimage.morphology import medial_axis
from sklearn.cluster import DBSCAN
import matplotlib.colors as colors
import matplotlib.cm as cmx
from sklearn.linear_model import LinearRegression
# https://www.titanwolf.org/Network/q/1ff55163-f706-463b-9b2d-0544f92612bf/y
# https://stackoverflow.com/questions/14184147/detect-lines-opencv-in-object
# https://www.programmersought.com/article/39844511557/
# aboout curvature:
# https://github.com/silenc3502/PyAdvancedLane#readme
def get_cmap(N):
    '''
    Returns a function that maps each index in 0, 1, ... N-1 to a distinct 
    RGB color.
    '''
    color_norm  = colors.Normalize(vmin=0, vmax=N-1)
    scalar_map = cmx.ScalarMappable(norm=color_norm, cmap='nipy_spectral') 
    def map_index_to_rgb_color(index):
        return scalar_map.to_rgba(index)
    return map_index_to_rgb_color

def doLineFitwithPolyfit(cluster, out_img,ikey,AngRet):

    rows,cols = out_img.shape[:2]
    x_select = cluster[:, 1]
    y_select = cluster[:, 0]

    plt.title('scattering of points of lane')
    # plt.scatter(x_select,y_select,marker = 'o')
    plt.plot(x_select, y_select, 'o', color='yellow')
    plt.xlim(0, cols)
    plt.ylim(rows, 0)
    plt.show()


    coef = np.polyfit(x_select,y_select,1)  # 1: rank = 1
    # print (coef[0])
    angle = np.rad2deg(np.arctan(1/coef[0]))      
    
    index='lane_'+str(ikey)
    AngRet[index] = angle
    print ('{0} AngleOffse = {1}'.format(index,AngRet[index]))        
    # y_pred = k*x + d
    poly1d_fn = np.poly1d(coef) 
    plt.plot(x_select,y_select, 'yo', x_select, poly1d_fn(x_select), '--k')
    # plt.xlim(0, columns)
    # plt.ylim(rows,0)
    plt.show()
    # pylab.scatter(lenx,datax)
    plt.savefig("/home/dom/ARWAC/prototype/Weed3DPrj/output/npPolyfit.png")
    # https://www.geeksforgeeks.org/plot-a-point-or-a-line-on-an-image-with-matplotlib/
    plt.plot(x_select, poly1d_fn(x_select), color="red", linewidth=3)
    plt.xlim(0, cols)
    plt.ylim(rows, 0)
    plt.show()
    plt.imshow(out_img)

    plt.show()        
    plt.savefig("/home/dom/ARWAC/prototype/Weed3DPrj/output/lincenters_over.png",bbox_inches="tight",pad_inches=0.02,dpi=250)
    return AngRet
        
def doLineFitwithCv2fitLine(cluster, warped,ikey,AngRet):

    rows = warped.shape[0]    
    cols = warped.shape[1] 
    [vx,vy,xx,yy] = cv2.fitLine(cluster, cv2.DIST_L2, 0, 0.01, 0.01)
    k = vy / vx
    b = yy - k * xx
    # k = output[1] / output[0]
    # b = output[3] - k * output[2]
    print ('k{0} = , b={1}'.format(k,b))
    #  Now find two extreme points on the line to draw line
    lefty = int((-xx*vy/vx) + yy)
    righty = int(((cols-xx)*vy/vx)+yy)
    # cv2.line(warped,(cols-1,righty),(0,lefty),(0,255,0),2)
    warped = cv2.line(warped,(lefty,rows-1),(righty,0),(0,0,255),4)
    cv2.imwrite("/home/dom/ARWAC/prototype/Weed3DPrj/output/cv2fitlines.png", warped)
    print('lefty = ', lefty, 'righty = ', righty)
    # https://www.titanwolf.org/Network/q/1ff55163-f706-463b-9b2d-0544f92612bf/y
    # https://stackoverflow.com/questions/14184147/detect-lines-opencv-in-object
    # https://www.programmersought.com/article/39844511557/
    index='lane_'+str(ikey)
    angle = np.rad2deg(np.arctan(k))   
    AngRet[index] = angle
    print ('{0} AngleOffset = {1}'.format(index,AngRet[index])) 
    """

    if abs(lefty)>cols or abs(righty)>cols:
        warped = cv2.line(warped,(xx,rows-1),(xx,0),(200,0,200),4)
    else:
        warped = cv2.line(warped,(lefty,rows-1),(righty,0),(0,0,255),4)
        #warped =  cv2.line(warped,(cols-1,righty),(0,lefty),(0,255,0),2)"""
    plt.title('linefit')
    plt.imshow(warped, cmap=plt.cm.gray)
    plt.show()
    return AngRet

def line_fitness(pts, image, color=(0, 0, 255)):
        h, w, ch = image.shape
        [vx, vy, x, y] = cv2.fitLine(np.array(pts), cv2.DIST_L1, 0, 0.01, 0.01)
        y1 = int((-x * vy / vx) + y)
        y2 = int(((w - x) * vy / vx) + y)
        cv2.line(image, (w - 1, y2), (0, y1), color, 2)
        return image
def manual_linedetector(dist,warped):
    h = warped.shape[0]    
    w = warped.shape[1]  
    result = np.zeros((h, w), dtype=np.uint8)
    ypts = []
    for row in range(h):
        cx = 0
        cy = 0
        max_d = 0
        for col in range(w):
            d = dist[row][col]
            if d > max_d:
                max_d = d
                cx = col
                cy = row
        result[cy][cx] = 255
        ypts.append([cx, cy])

    xpts = []
    for col in range(w):
        cx = 0
        cy = 0
        max_d = 0
        for row in range(h):
            d = dist[row][col]
            if d > max_d:
                max_d = d
                cx = col
                cy = row
        result[cy][cx] = 255
        xpts.append([cx, cy])


    plt.imshow(result)
    plt.show()
    cv2.imwrite("/home/dom/Documents/ARWAC/robdata/lineSegs/skeleton.png", result)
    plt.close()

    frame = line_fitness(ypts, image=warped)
    frame = line_fitness(xpts, image=frame, color=(255, 0, 0))

    plt.imshow(frame)
    plt.show()
    cv2.imwrite("/home/dom/Documents/ARWAC/robdata/lineSegs/fitlines.png", frame)
    plt.close()

def get_lanes_dbscan(mask_bin,warped,eps=45):      
    
    rows = mask_bin.shape[0]    
    cols = mask_bin.shape[1]  

    mask_bin[mask_bin < 0] = 0  # this is optional and can be skipped to handle any threshold other than 'non zero value' & 'zero'
    img = mask_bin.astype(np.uint8)
    print(f"sparsity: {len(mask_bin[mask_bin != 0]) * 100 / np.cumproduct(mask_bin.shape)[-1]} %")
    plt.figure('img',figsize=(6,4))
    plt.title('img')
    plt.imshow(img,cmap = 'gray', vmin=0, vmax=1)
    plt.show(block=~True)
    
    non_zero_coords=np.nonzero(img)  # output tuple
    X =  np.transpose(non_zero_coords)


    if len(X) ==0:
        return None

    db = DBSCAN(eps=eps, min_samples=64).fit(X)  # eps = 0.3//// 2 or 10 all OK, 3 
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    cluster_labels = db.labels_
    # Black removed and is used for noise instead.
    unique_labels = set(cluster_labels)
    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(unique_labels) - (1 if -1 in cluster_labels else 0)
    n_noise_ = list(cluster_labels).count(-1)

    print('Estimated number of clusters: %d' % n_clusters_)   
    
    # n_clusters = len(unique_labels)
    colors = [plt.cm.Spectral(each)
            for each in np.linspace(0, 1, n_clusters_)]
            
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = (cluster_labels == k)

        xy = X[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 1], xy[:, 0], 'o', markerfacecolor=tuple(col),
                markeredgecolor='k', markersize=4)

        xy = X[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 1], xy[:, 0], 'o', markerfacecolor=tuple(col),
                markeredgecolor='k', markersize=6)

    plt.title('Estimated number of clusters: %d' % n_clusters_)
    plt.show()
    lane_dic = [X[cluster_labels == n] for n in range(0, n_clusters_)]

    # clusters = [X[unique_labels == n] for n in range(0, n_clusters_)]
    # cmaps = get_cmap(n_clusters_)
    plt.figure(figsize = (8, 4))
    fig = plt.gcf()
    fig.canvas.set_window_title('DBSCAN Cluster...')
    regr = LinearRegression()
    AngRet = {}
    result = np.zeros((rows, cols), dtype=np.uint8)
    ypts = []
    dist = mask_bin / 1

    # manual processinn........... https://www.programmersought.com/article/66584665663/
    # manual_linedetector(dist,warped)
    for ikey, cluster in enumerate(lane_dic):

        # try linear regression, not working  as multirate data sets
        # regr = LinearRegression()
        # using np.polyfit
        # AngRet = doLineFitwithPolyfit(cluster, warped,ikey,AngRet)
        
        # Cv2.fitline
        # AngRet = doLineFitwithCv2fitLine(cluster, warped,ikey,AngRet)    
        # 
        # # using np.polyfit 
        coef = doLineFitwithskLearRegress(regr,cluster, warped,ikey)
        index='lane_'+str(ikey)
        angle = np.rad2deg(np.arctan(coef))   
        AngRet[index] = angle
        print ('{0} AngleOffset = {1}'.format(index,AngRet[index]))

    # draw_display_oldfashion(out_img,lane_dic)
    # draw_display_dbscan(cluster_labels,n_clusters,out_img,labels,core_samples_mask, X)
    ####################################################
    ####################################################
    # return non_zero_coords,out_img,mask_bin,eps
    return AngRet, lane_dic
def doLineFitwithskLearRegress(regr,cluster, warped,ikey):

    rows,cols = warped.shape[:2]
    x_select = cluster[:, 1]
    y_select = cluster[:, 0]
    plt.title('scattering of points of lane')
    plt.scatter(x_select,y_select)
    plt.show()
    # plt.close()
    # plt.savefig("/home/dom/ARWAC/prototype/Weed3DPrj/output/sklincenters.png")
    plt.savefig("/home/dom/Documents/ARWAC/robdata/lineSegs/sklincenters.png")


    x = np.array(x_select).reshape((-1, 1))
    y = np.array(y_select)
    regr.fit(x, y)
    r_sq = regr.score(x, y)
    print('intercept:', regr.intercept_)
    print('slope:', regr.coef_)
    plt.plot(x, regr.predict(x), color='#52b920', label='Regression Line')
        # Ploting Scatter Points
    plt.scatter(x_select, y_select, c='#ef4423', label='Scatter Plot')
    # plt.xticks(())
    # plt.yticks(())
    plt.imshow(warped)
    plt.xlabel('x')
    plt.ylabel('y ')
    plt.legend()
    plt.show()
    # plt.savefig("/home/dom/ARWAC/prototype/Weed3DPrj/output/sklineScatters.png")
    plt.savefig("/home/dom/Documents/ARWAC/robdata/lineSegs/sklineScatters.png")
    return regr.coef_

def GetStreeingInfo(mask_ske,warped,eps):
    AngRet, lane_dic = get_lanes_dbscan(mask_ske,warped,eps)
    return AngRet, lane_dic
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
    ret1,th1 = cv2.threshold(gscale,128,255,cv2.THRESH_BINARY)
    skel, distance = medial_axis(th1, return_distance=True)
    # Distance to the background for pixels of the skeleton
    mask_ske = distance * skel
    # using cv2. built in function
    # https://www.programmersought.com/article/66584665663/
    # dist = cv2.distanceTransform(th1, cv2.DIST_L1, cv2.DIST_MASK_PRECISE)

    # dist = cv2.distanceTransform(th1, cv2.DIST_L1, 3)
    # dist = dist / 15

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
    fig.savefig(fullpth, dpi=None, facecolor='w', edgecolor='w',orientation='portrait', papertype=None, format=None,transparent=False, bbox_inches=None, pad_inches=0.1, frameon=None, metadata=None)
 
    plt.show(block=True)
    plt.pause(0.25)
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