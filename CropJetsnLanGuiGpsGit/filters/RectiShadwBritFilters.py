'''
Class: ParticleFilter 
implements simple particle filter algorithm.
Author: Vinay
me@vany.in
'''
import numpy as np
import scipy
import scipy.stats
from numpy.random import uniform,randn
from numpy.linalg import norm
from skimage import io, color
import matplotlib.pyplot as plt
from PIL import  Image
# LIBRARIES
#############
import cv2
import os
import sys
import matplotlib.image as mpimg
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
# from filterpy.monte_carlo import systematic_resample


class AllFilters:
	def __init__(self,N=1000,x_range = (0,800),sensor_err=1,par_std=100):
		self.N = N
		self.x_range = x_range
		# self.create_uniform_particles()
		self.weights = np.zeros(N)
		self.u = 0.00
		self.initial_pose = 0#lane_id
		self.sensor_std_err = sensor_err
		self.results = {}


	def ShadowRemove_EntropyFinlayson(self, img):
		self.results = {}
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		# https://www.tfzx.net/index.php/article/5931706.html
		h, w = img.shape[:2]

		sz = 1

		while sz<99:
			
			result_planes = []
			result_norm_planes = []
			sz=sz+2
			print('basic filter size = {0} and Median filter size = {1}'.format(sz,0))

			img = cv2.GaussianBlur(img, (sz,sz), 0)

			# Separate Channels
			r, g, b = cv2.split(img) 
			im_sum = np.sum(img, axis=2)
			im_mean = gmean(img, axis=2)
			# Create "normalized", mean, and rg chromaticity vectors
			#  We use mean (works better than norm). rg Chromaticity is
			#  for visualization
			n_r = np.ma.divide( 1.*r, g )
			n_b = np.ma.divide( 1.*b, g )

			mean_r = np.ma.divide(1.*r, im_mean)
			mean_g = np.ma.divide(1.*g, im_mean)
			mean_b = np.ma.divide(1.*b, im_mean)

			rg_chrom_r = np.ma.divide(1.*r, im_sum)
			rg_chrom_g = np.ma.divide(1.*g, im_sum)
			rg_chrom_b = np.ma.divide(1.*b, im_sum)

			# Visualize rg Chromaticity --> DEBUGGING
			rg_chrom = np.zeros_like(img)

			rg_chrom[:,:,0] = np.clip(np.uint8(rg_chrom_r*255), 0, 255)
			rg_chrom[:,:,1] = np.clip(np.uint8(rg_chrom_g*255), 0, 255)
			rg_chrom[:,:,2] = np.clip(np.uint8(rg_chrom_b*255), 0, 255)

			plt.imshow(rg_chrom)
			plt.title('rg Chromaticity')
			plt.show()

			#-----------------------
			## 2. Take Logarithms ##
			#-----------------------

			l_rg = np.ma.log(n_r)
			l_bg = np.ma.log(n_b)

			log_r = np.ma.log(mean_r)
			log_g = np.ma.log(mean_g)
			log_b = np.ma.log(mean_b)

			##  rho = np.zeros_like(img, dtype=np.float64)
			##
			##  rho[:,:,0] = log_r
			##  rho[:,:,1] = log_g
			##  rho[:,:,2] = log_b

			rho = cv2.merge((log_r, log_g, log_b))

			# Visualize Logarithms --> DEBUGGING
			plt.scatter(l_rg, l_bg, s = 2)
			plt.xlabel('Log(R/G)')
			plt.ylabel('Log(B/G)')
			plt.title('Log Chromaticities')
			plt.show()

			plt.scatter(log_r, log_b, s = 2)
			plt.xlabel('Log( R / 3root(R*G*B) )')
			plt.ylabel('Log( B / 3root(R*G*B) )')
			plt.title('Geometric Mean Log Chromaticities')
			plt.show()

			#----------------------------
			## 3. Rotate through Theta ##
			#----------------------------
			u = 1./np.sqrt(3)*np.array([[1,1,1]]).T
			I = np.eye(3)

			tol = 1e-15

			P_u_norm = I - u.dot(u.T)
			U_, s, V_ = np.linalg.svd(P_u_norm, full_matrices = False)

			s[ np.where( s <= tol ) ] = 0.

			U = np.dot(np.eye(3)*np.sqrt(s), V_)
			U = U[ ~np.all( U == 0, axis = 1) ].T

			# Columns are upside down and column 2 is negated...?
			U = U[::-1,:]
			U[:,1] *= -1.

			##  TRUE ARRAY:
			##
			##  U = np.array([[ 0.70710678,  0.40824829],
			##                [-0.70710678,  0.40824829],
			##                [ 0.        , -0.81649658]])

			chi = rho.dot(U) 

			# Visualize chi --> DEBUGGING
			plt.scatter(chi[:,:,0], chi[:,:,1], s = 2)
			plt.xlabel('chi1')
			plt.ylabel('chi2')
			plt.title('2D Log Chromaticities')
			plt.show()

			e = np.array([[np.cos(np.radians(np.linspace(1, 180, 180))), \
						np.sin(np.radians(np.linspace(1, 180, 180)))]])

			gs = chi.dot(e)

			prob = np.array([np.histogram(gs[...,i], bins='scott', density=True)[0] 
							for i in range(np.size(gs, axis=3))])

			eta = np.array([entropy(p, base=2) for p in prob])

			plt.plot(eta)
			plt.xlabel('Angle (deg)')
			plt.ylabel('Entropy, eta')
			plt.title('Entropy Minimization')
			plt.show()

			theta_min = np.radians(np.argmin(eta))

			print('Min Angle: ', np.degrees(theta_min))

			e = np.array([[-1.*np.sin(theta_min)],
						[np.cos(theta_min)]])

			gs_approx = chi.dot(e)

			# Visualize Grayscale Approximation --> DEBUGGING
			plt.imshow(gs_approx.squeeze(), cmap='gray')
			plt.title('Grayscale Approximation')
			plt.show()

			P_theta = np.ma.divide( np.dot(e, e.T), np.linalg.norm(e) )

			chi_theta = chi.dot(P_theta)
			rho_estim = chi_theta.dot(U.T)
			mean_estim = np.ma.exp(rho_estim)

			estim = np.zeros_like(mean_estim, dtype=np.float64)

			estim[:,:,0] = np.divide(mean_estim[:,:,0], np.sum(mean_estim, axis=2))
			estim[:,:,1] = np.divide(mean_estim[:,:,1], np.sum(mean_estim, axis=2))
			estim[:,:,2] = np.divide(mean_estim[:,:,2], np.sum(mean_estim, axis=2))

			plt.imshow(estim)
			plt.title('Invariant rg Chromaticity')
			plt.show()

			# res = np.hstack((result,result_norm))
			# self.PltShowing(res,'shadwRemove_adptive', 'after filtering...')		
			
		self.results[1] = gs_approx
		self.results[2] = estim

		return self.results

	def Shdw1DIlluminantInvartImg(self,img):
		# https://github.com/srijan-mishra/Shadow-Removal/blob/master/1D%20Illuminant%20invariant%20image.py
		# img=cv2.imread('/media/dom/Elements/Color_Shadow/Shadow-Removal/images/Successful Cases/2.png') #path to the image
		
		img=np.float64(img)

		blue,green,red=cv2.split(img)

		blue[blue==0]=1
		green[green==0]=1
		red[red==0]=1

		div=np.multiply(np.multiply(blue,green),red)**(1.0/3)

		a=np.log1p((blue/div)-1)
		b=np.log1p((green/div)-1)
		c=np.log1p((red/div)-1)

		a1 = np.atleast_3d(a)
		b1 = np.atleast_3d(b)
		c1 = np.atleast_3d(c)
		rho= np.concatenate((c1,b1,a1),axis=2) #log chromaticity on a plane

		U=[[1/math.sqrt(2),-1/math.sqrt(2),0],[1/math.sqrt(6),1/math.sqrt(6),-2/math.sqrt(6)]]
		U=np.array(U) #eigens

		X=np.dot(rho,U.T) #2D points on a plane orthogonal to [1,1,1]


		d1,d2,d3=img.shape

		e_t=np.zeros((2,181))
		for j in range(181):
			e_t[0][j]=math.cos(j*math.pi/180.0)
			e_t[1][j]=math.sin(j*math.pi/180.0)

		Y=np.dot(X,e_t)
		nel=img.shape[0]*img.shape[1]

		bw=np.zeros((1,181))

		for i in range(181):
			bw[0][i]=(3.5*np.std(Y[:,:,i]))*((nel)**(-1.0/3))

		entropy=[]
		for i in range(181):
			temp=[]
			comp1=np.mean(Y[:,:,i])-3*np.std(Y[:,:,i])
			comp2=np.mean(Y[:,:,i])+3*np.std(Y[:,:,i])
			for j in range(Y.shape[0]):
				for k in range(Y.shape[1]):
					if Y[j][k][i]>comp1 and Y[j][k][i]<comp2:
						temp.append(Y[j][k][i])
			nbins=round((max(temp)-min(temp))/bw[0][i])
			nbins = int (nbins)
			(hist,waste)=np.histogram(temp,bins=nbins)
			hist=filter(lambda var1: var1 != 0, hist)
			hist1=np.array([float(var) for var in hist])
			hist1=hist1/sum(hist1)
			entropy.append(-1*sum(np.multiply(hist1,np.log2(hist1))))

		angle=entropy.index(min(entropy))

		e_t=np.array([math.cos(angle*math.pi/180.0),math.sin(angle*math.pi/180.0)])
		e=np.array([-1*math.sin(angle*math.pi/180.0),math.cos(angle*math.pi/180.0)])

		I1D=np.exp(np.dot(X,e_t)) #mat2gray to be done


		p_th=np.dot(e_t.T,e_t)
		X_th=X*p_th
		mX=np.dot(X,e.T)
		mX_th=np.dot(X_th,e.T)

		mX=np.atleast_3d(mX)
		mX_th=np.atleast_3d(mX_th)

		theta=(math.pi*float(angle))/180.0
		theta=np.array([[math.cos(theta),math.sin(theta)],[-1*math.sin(theta),math.cos(theta)]])
		alpha=theta[0,:]
		alpha=np.atleast_2d(alpha)
		beta=theta[1,:]
		beta=np.atleast_2d(beta)




		#Finding the top 1% of mX
		mX1=mX.reshape(mX.shape[0]*mX.shape[1])
		mX1sort=np.argsort(mX1)[::-1]
		mX1sort=mX1sort+1
		mX1sort1=np.remainder(mX1sort,mX.shape[1])
		mX1sort1=mX1sort1-1
		mX1sort2=np.divide(mX1sort,mX.shape[1])
		mX_index=[[x,y,0] for x,y in zip(list(mX1sort2),list(mX1sort1))]
		mX_top=[mX[x[0],x[1],x[2]] for x in mX_index[:int(0.01*mX.shape[0]*mX.shape[1])]]
		mX_th_top=[mX_th[x[0],x[1],x[2]] for x in mX_index[:int(0.01*mX_th.shape[0]*mX_th.shape[1])]]
		X_E=(statistics.median(mX_top)-statistics.median(mX_th_top))*beta.T
		X_E=X_E.T

		for i in range(X_th.shape[0]):
			for j in range(X_th.shape[1]):
				X_th[i,j,:]=X_th[i,j,:]+X_E

		rho_ti=np.dot(X_th,U)
		c_ti=np.exp(rho_ti)
		sum_ti=np.sum(c_ti,axis=2)
		sum_ti=sum_ti.reshape(c_ti.shape[0],c_ti.shape[1],1)
		r_ti=c_ti/sum_ti

		r_ti2=255*r_ti

		plt.imshow(r_ti2)
		plt.title('1D invariant ')
		plt.show()
		self.results[1] = img
		self.results[2] = r_ti2

		return self.results
		# cv2.imwrite('p003-1.png',r_ti2) #path to directory where image is saved


	def PltShowing(self,img,figname,titlname, pauseflg=True, dipFlag = True):
		# return
		# paseflg=False
		if dipFlag == True:
			plt.figure(figname,figsize=(36,24))
			plt.title(titlname)
			plt.imshow(img,cmap='gray')
			# vmin=0, vmax=1)    for colorscale : the value ranage of colorscale of each map
			# plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB),cmap='gray', vmin=0, vmax=1)   
			plt.show(block=pauseflg)  
			plt.pause(0.25)
			plt.close()
		else:
			return

	def nothing(self,x):
		pass
	def brightness1(self,img):
		"""returns bright ness processed  with opencv method"""
		self.results = {}
		# cv2.namedWindow('image')
		# cv2.createTrackbar('val', 'image', 100, 150, self.nothing)
		# https://github.com/spmallick/learnopencv/blob/master/Photoshop-Filters-in-OpenCV/brightness.py
		# https://www.learnopencv.com/photoshop-filters-in-opencv/?ck_subscriber_id=272185985
		index = 0
		while index<150:
			hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
			hsv = np.array(hsv, dtype=np.float64)
			# val = cv2.getTrackbarPos('val', 'image')
			val = index/100 # dividing by 100 to get in range 0-1.5
			# scale pixel values up or down for channel 1(Saturation)
			hsv[:, :, 1] = hsv[:, :, 1] * val
			hsv[:, :, 1][hsv[:, :, 1] > 255] = 255 # setting values > 255 to 255.
			# scale pixel values up or down for channel 2(Value)
			hsv[:, :, 2] = hsv[:, :, 2] * val
			hsv[:, :, 2][hsv[:, :, 2] > 255] = 255 # setting values > 255 to 255.
			hsv = np.array(hsv, dtype=np.uint8)
			res = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

			self.PltShowing(res,'Brightness adjust', 'after...')
			# cv2.imshow("original", img)
			# cv2.imshow('image', res)		
			print('val = ', val)	

			# if cv2.waitKey(1) & 0xFF == ord('q'):
			index=index+1
			self.results[1] = img
			self.results[2] = res

				# break
			# cv2.destroyAllWindows()		
		return self.results 

	def shadwRemove_adptiv(self,img):
        
		self.results = {}
		rgb_planes = cv2.split(img)
		# cv2.namedWindow('image')
		# cv2.createTrackbar('val', 'image', 100, 150, self.nothing)
		# https://github.com/spmallick/learnopencv/blob/master/Photoshop-Filters-in-OpenCV/brightness.py
		# https://www.learnopencv.com/photoshop-filters-in-opencv/?ck_subscriber_id=272185985
		sz = 1
		while sz<99:
			
			result_planes = []
			result_norm_planes = []
			sz=sz+2
			for plane in rgb_planes:
				dilated_img = cv2.dilate(plane, np.ones((sz,sz), np.uint8))

				sz_med = sz*3
				if sz>33:
					sz_med = 99		
				print('basic filter size = {0} and Median filter size = {1}'.format(sz,sz_med))
				bg_img = cv2.medianBlur(dilated_img, sz_med)
				diff_img = 255 - cv2.absdiff(plane, bg_img)
				norm_img = cv2.normalize(diff_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
				result_planes.append(diff_img)
				result_norm_planes.append(norm_img)
            
			result = cv2.merge(result_planes)
			result_norm = cv2.merge(result_norm_planes)
			res = np.hstack((result,result_norm))
			self.PltShowing(res,'shadwRemove_adptive', 'after filtering...')		
			
		self.results[1] = result
		self.results[2] = result_norm

		return self.results

	def shadwRemove1(self,img, svepflg = True, indexes = False):
		""" Opencv default methods """
		self.results = {}		
		rgb_planes = cv2.split(img)

		result_planes = []
		result_norm_planes = []
		for plane in rgb_planes:
			dilated_img = cv2.dilate(plane, np.ones((7,7), np.uint8))
			bg_img = cv2.medianBlur(dilated_img, 21)
			diff_img = 255 - cv2.absdiff(plane, bg_img)
			norm_img = cv2.normalize(diff_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
			result_planes.append(diff_img)
			result_norm_planes.append(norm_img)

		result = cv2.merge(result_planes)
		result_norm = cv2.merge(result_norm_planes)

		if svepflg==True:
			cv2.imwrite('shadows_out.png', result)
			cv2.imwrite('shadows_out_norm.png', result_norm)
			
		self.results[1] = result
		self.results[2] = result_norm

		return self.results
	
	def shadowremoval_YCbCr(self, or_img):
		# https://github.com/mykhailo-mostipan/shadow-removal/blob/master/shadow-removal-based-on-YCbCr-color-space.py
		self.results = {}
		sz = 1
		while sz<99:
			sz=sz+2
			print('basic filter size = {0} and Median filter size = {1}'.format(sz,0))
			# covert the BGR image to an YCbCr image
			y_cb_cr_img = cv2.cvtColor(or_img, cv2.COLOR_BGR2YCrCb)
			# copy the image to create a binary mask later
			binary_mask = np.copy(y_cb_cr_img)
			# get mean value of the pixels in Y plane
			y_mean = np.mean(cv2.split(y_cb_cr_img)[0])
			# get standard deviation of channel in Y plane
			y_std = np.std(cv2.split(y_cb_cr_img)[0])
			# classify pixels as shadow and non-shadow pixels
			for i in range(y_cb_cr_img.shape[0]):
				for j in range(y_cb_cr_img.shape[1]):
					if y_cb_cr_img[i, j, 0] < y_mean - (y_std / 3):
						# paint it white (shadow)
						binary_mask[i, j] = [255, 255, 255]
					else:
						# paint it black (non-shadow)
						binary_mask[i, j] = [0, 0, 0]

			# Using morphological operation
			# The misclassified pixels are
			# removed using dilation followed by erosion.
			kernel = np.ones((sz, sz), np.uint8)
			erosion = cv2.erode(binary_mask, kernel, iterations=1)

			# sum of pixel intensities in the lit areas
			spi_la = 0
			# sum of pixel intensities in the shadow
			spi_s = 0
			# number of pixels in the lit areas
			n_la = 0
			# number of pixels in the shadow
			n_s = 0
			# get sum of pixel intensities in the lit areas
			# and sum of pixel intensities in the shadow
			for i in range(y_cb_cr_img.shape[0]):
				for j in range(y_cb_cr_img.shape[1]):
					if erosion[i, j, 0] == 0 and erosion[i, j, 1] == 0 and erosion[i, j, 2] == 0:
						spi_la = spi_la + y_cb_cr_img[i, j, 0]
						n_la += 1
					else:
						spi_s = spi_s + y_cb_cr_img[i, j, 0]
						n_s += 1

			# get the average pixel intensities in the lit areas
			average_ld = spi_la / n_la
			# get the average pixel intensities in the shadow
			average_le = spi_s / n_s
			# difference of the pixel intensities in the shadow and lit areas
			i_diff = average_ld - average_le
			# get the ratio between average shadow pixels and average lit pixels
			ratio_as_al = average_ld / average_le
			# added these difference
			for i in range(y_cb_cr_img.shape[0]):
				for j in range(y_cb_cr_img.shape[1]):
					if erosion[i, j, 0] == 255 and erosion[i, j, 1] == 255 and erosion[i, j, 2] == 255:

						y_cb_cr_img[i, j] = [y_cb_cr_img[i, j, 0] + i_diff, y_cb_cr_img[i, j, 1] + ratio_as_al,
											y_cb_cr_img[i, j, 2] + ratio_as_al]

			# covert the YCbCr image to the BGR image
			final_image = cv2.cvtColor(y_cb_cr_img, cv2.COLOR_YCR_CB2BGR)

			# res = np.hstack((result,result_norm))
			self.PltShowing(final_image,'shadow-removal-based-on-YCbCr-color-space', 'after filtering...')		
			
		self.results[1] = or_img
		self.results[2] = final_image

		return self.results

if __name__ == '__main__':
	print ( "all filers for the image manipulation class implementation")
	xl_int_pf=AllFilters(N=10000,x_range=(0,800),ses_err=1,par_std=100)
