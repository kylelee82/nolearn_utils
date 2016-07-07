#!C:\Miniconda2\python.exe

#################
#
# Description: Equivalent to Binary_JPG_Classifier, but this version uses Theano/Lasagne/Nolearn for CNN implementation rather than standard RF or SVM
#
# Idea: Improved object detection implementation.  Obstructions like clothing should be NO problem
#
# Recommend at least 1000 images per class for proper analysis
#
#################
from scipy import misc
from scipy import ndimage
import random
import numpy as np
from nolearn.lasagne import PrintLayerInfo
from lasagne import layers
from lasagne.updates import nesterov_momentum
from lasagne.updates import adadelta
from lasagne.updates import sgd
from lasagne.updates import adam
from lasagne.nonlinearities import softmax
from nolearn.lasagne import NeuralNet
from nolearn.lasagne import BatchIterator
from nolearn.lasagne import TrainSplit
import matplotlib.pyplot as plt
import glob
from numpy import *
import os
import time
import pickle
import cPickle
import sys
import theano
import cv2
import lasagne
from sklearn import cross_validation
from lasagne.layers import set_all_param_values
from lasagne.layers import InputLayer
from lasagne.layers import Conv2DLayer
from lasagne.layers import Pool2DLayer
from lasagne.layers import DenseLayer
from lasagne.layers import GlobalPoolLayer
from lasagne.layers import ConcatLayer
from lasagne.layers import DilatedConv2DLayer
from lasagne.layers.normalization import batch_norm
from lasagne.nonlinearities import softmax, rectify
from lasagne.layers import NonlinearityLayer
from lasagne.layers import ExpressionLayer
from lasagne.layers import PadLayer
from lasagne.layers import GlobalPoolLayer
from lasagne.layers import ElemwiseSumLayer
from lasagne.layers import Conv2DLayer as ConvLayer
from lasagne.layers import DropoutLayer
from lasagne.layers import Upscale2DLayer
import re
from skimage.transform import resize
from skimage.transform import SimilarityTransform
from skimage.transform import AffineTransform
#from skimage.transform import warp
from skimage.transform._warps_cy import _warp_fast
from lasagne.nonlinearities import sigmoid
import theano.tensor as T

class CNN_JPG_Classifier:

    X_train = None
    Y_train = None
    X_cv    = None
    Y_cv    = None
    classifier = None
    n_train_max = 30000
    grayscale = True
    resize    = False

    # autogate (full width)
    # idea is to detect EVEN IF clothes are there (with obstructions)
    img_y1     = 45
    img_y2     = 290
    img_x1     = 165
    img_x2     = 410
    img_width  = (img_x2 - img_x1)
    img_height = (img_y2 - img_y1)
    img_ch     = 1
    augmented  = True
    zoom_mode  = True
    augmented2 = False
    pretrain_model = False
    pretrain_name = "default"

    # num_classes for multi-class support
    num_class  = 2
    
    # for data augmentation
    img_shift  = 10

    # VGG16 average
    # IMAGE_MEAN = np.array([103.939,116.779,123.68]).reshape(3,1,1)
    # VGG_CNN_S average
    IMAGE_MEAN = np.array([102.717,115.773,123.51]).reshape(3,1,1)

    verbose = False
    
    def __init__(self):
        self.classifier = None

    def set_verbosity(self,boolean):
        self.verbose = boolean

    # support color implementation for natural images
    def set_grayscale(self,bool):
        self.grayscale = bool
        if bool is True:
            self.img_ch = 1
        else:
            self.img_ch = 3

    def augment_training(self,bool):
        self.augmented = bool

    def augment_training2(self,bool):
	self.augmented2 = bool

    def set_pretrain_model(self,bool):
	self.pretrain_model = bool

    def set_pretrain_name(self,name):
	self.pretrain_name = name

    def set_zoom_mode(self,bool):
	self.zoom_mode = bool
            
    # support resize mode so that images are shrunk (64x64 is a good number)
    def set_resize(self,bool,width,height):
        self.resize = bool
        self.img_width = width
        self.img_height = height

    def set_img_zoom(self,y1,y2,x1,x2):
        self.img_y1 = y1
        self.img_y2 = y2
        self.img_x1 = x1
        self.img_x2 = x2
        self.img_width = abs(x2-x1)
        self.img_height = abs(y2-y1)

    def resize_image(self,img):
        if self.resize is True:
            return misc.imresize(img,[self.img_width,self.img_height])
        else:
            return img

    def set_num_classes(self,class_cnt):
        self.num_class = class_cnt;

    def set_max_train(self,max_train):
	self.n_train_max = max_train

    # training - read and save to cache
    def read_train_dirs(self,close_dir,open_dir,extra=list(),type="jpg",blur=True):
        closed_gate_pics = list(glob.iglob(close_dir+"/*."+type))
        open_gate_pics   = list(glob.iglob(open_dir+"/*."+type))

        # for multi-class support
        extra_len = 0
        extra_pics_listoflist = list()
        if self.num_class > 2:
            for i in range(2,self.num_class):
                print("Extra class:"+extra[i-2])
                extra_pics_listoflist.append(list(glob.iglob(extra[i-2]+"/*."+type)))
                extra_len += len(list(glob.iglob(extra[i-2]+"/*."+type)))
        multiplier = 1
        if self.augmented is True:
            multiplier = 4
        if self.augmented2 is True:
	    multiplier = 3

        nsamples = min(min(len(closed_gate_pics),ceil(self.n_train_max/self.num_class)) + min(len(open_gate_pics),ceil(self.n_train_max/self.num_class)) + min(extra_len,ceil((self.num_class - 2) * self.n_train_max/self.num_class)),self.n_train_max)
        print("Number of samples, expected:"+str(nsamples))
        print("Number of samples, closed:"+str(min(len(closed_gate_pics),ceil(self.n_train_max/self.num_class))))
        print("Number of samples, open:"+str(min(len(open_gate_pics),ceil(self.n_train_max/self.num_class))))        
        print("Number of samples, others:"+str(min(extra_len,ceil((self.num_class - 2) * self.n_train_max/self.num_class))))               
        if self.grayscale is True:
            inputs  = np.zeros(shape=(nsamples*multiplier,self.img_height,self.img_width)) # only at the section of interest
        else:
            inputs  = np.zeros(shape=(nsamples*multiplier,3,self.img_height,self.img_width)) # RGB
        targets = np.zeros(shape=(nsamples*multiplier,)) # vector only
        i=0
        for pic in closed_gate_pics:
            if i < multiplier * self.n_train_max / self.num_class:
                if self.verbose == True:
                    print("Reading (training) class 0 " + str(i) + ":"+pic)
                img = misc.imread(pic)
                if self.grayscale is True:
                    if img.shape == (self.img_width, self.img_height):
                        inputs[i,:,:] = img[:,:]
                    else:
			if blur is True:
	                        img_blur = ndimage.gaussian_filter(img,sigma=0.1)                    
			else:
				img_blur = img
			if self.zoom_mode is True:
	                        inputs[i,:,:] = self.resize_image(self.rgb2gray(img_blur[self.img_y1:self.img_y2,self.img_x1:self.img_x2]))
			else:
	                        inputs[i,:,:] = self.resize_image(self.rgb2gray(img_blur))
                else:
                    img_blur = img
		    if self.zoom_mode is True:
	                    inputs[i,0,:,:] = self.resize_image(img_blur[self.img_y1:self.img_y2,self.img_x1:self.img_x2,0])
        	            inputs[i,1,:,:] = self.resize_image(img_blur[self.img_y1:self.img_y2,self.img_x1:self.img_x2,1])
                	    inputs[i,2,:,:] = self.resize_image(img_blur[self.img_y1:self.img_y2,self.img_x1:self.img_x2,2])
		    else:                   
	                    inputs[i,0,:,:] = self.resize_image(img_blur[:,:,0])
        	            inputs[i,1,:,:] = self.resize_image(img_blur[:,:,1])
                	    inputs[i,2,:,:] = self.resize_image(img_blur[:,:,2])

                targets[i] = 1

                if self.augmented is True:
                    if self.img_x1-self.img_shift >=0.0:
                        tmp_shift = self.img_shift
                    else:
                        tmp_shift = 0.0
                    if self.grayscale is True:
                        inputs[i+1,:,:] = self.resize_image(self.rgb2gray(img_blur[self.img_y1:self.img_y2,self.img_x1-tmp_shift:self.img_x2-tmp_shift]))
                    else:
                        inputs[i+1,0,:,:] = self.resize_image(img_blur[self.img_y1:self.img_y2,self.img_x1-tmp_shift:self.img_x2-tmp_shift,0])
                        inputs[i+1,1,:,:] = self.resize_image(img_blur[self.img_y1:self.img_y2,self.img_x1-tmp_shift:self.img_x2-tmp_shift,1])
                        inputs[i+1,2,:,:] = self.resize_image(img_blur[self.img_y1:self.img_y2,self.img_x1-tmp_shift:self.img_x2-tmp_shift,2])
                    targets[i+1] = 1
                    if self.img_x2+self.img_shift <= 640.0:
                        tmp_shift = self.img_shift
                    else:
                        tmp_shift = 0.0
                    if self.grayscale is True:
                        inputs[i+2,:,:] = self.resize_image(self.rgb2gray(img_blur[self.img_y1:self.img_y2,self.img_x1+tmp_shift:self.img_x2+tmp_shift]))
                    else:
                        inputs[i+2,0,:,:] = self.resize_image(img_blur[self.img_y1:self.img_y2,self.img_x1+tmp_shift:self.img_x2+tmp_shift,0])
                        inputs[i+2,1,:,:] = self.resize_image(img_blur[self.img_y1:self.img_y2,self.img_x1+tmp_shift:self.img_x2+tmp_shift,1])
                        inputs[i+2,2,:,:] = self.resize_image(img_blur[self.img_y1:self.img_y2,self.img_x1+tmp_shift:self.img_x2+tmp_shift,2])                   
                    targets[i+2] = 1
                    if self.img_y2+self.img_shift <= 480.0:
                        tmp_shift = self.img_shift
                    else:
                        tmp_shift = 0.0
                    if self.grayscale is True:
                        inputs[i+3,:,:] = self.resize_image(self.rgb2gray(img_blur[self.img_y1+tmp_shift:self.img_y2+tmp_shift,self.img_x1:self.img_x2]))
                    else:
                        inputs[i+3,0,:,:] = self.resize_image(img_blur[self.img_y1+tmp_shift:self.img_y2+tmp_shift,self.img_x1:self.img_x2,0])
                        inputs[i+3,1,:,:] = self.resize_image(img_blur[self.img_y1+tmp_shift:self.img_y2+tmp_shift,self.img_x1:self.img_x2,1])
                        inputs[i+3,2,:,:] = self.resize_image(img_blur[self.img_y1+tmp_shift:self.img_y2+tmp_shift,self.img_x1:self.img_x2,2])
                    targets[i+3] = 1
                    i+=4
                    #if self.verbose == True:
                    #    print("Reading (training) augmented closed gate pic " + str(i) + ":"+pic)
                else:
                    i+=1
        last_index = i
        for pic in open_gate_pics:
            if i - last_index < multiplier * self.n_train_max / self.num_class:
                if self.verbose == True:
                    print("Reading (training) class 1 "+str(i)+":"+pic)
                img = misc.imread(pic)
                if self.grayscale is True:
                    if img.shape == (self.img_width, self.img_height):
                        inputs[i,:,:] = img[:,:]
                    else:
                        if blur is True:
                        	img_blur = ndimage.gaussian_filter(img,sigma=0.1)
			else:
				img_blur = img
                        if self.zoom_mode is True:
	                        inputs[i,:,:] = self.resize_image(self.rgb2gray(img_blur[self.img_y1:self.img_y2,self.img_x1:self.img_x2]))
			else:
				inputs[i,:,:] = self.resize_image(self.rgb2gray(img_blur))
                else:
                    img_blur = img
                    if self.zoom_mode is True:
                            inputs[i,0,:,:] = self.resize_image(img_blur[self.img_y1:self.img_y2,self.img_x1:self.img_x2,0])
                            inputs[i,1,:,:] = self.resize_image(img_blur[self.img_y1:self.img_y2,self.img_x1:self.img_x2,1])
                            inputs[i,2,:,:] = self.resize_image(img_blur[self.img_y1:self.img_y2,self.img_x1:self.img_x2,2])
                    else:
                            inputs[i,0,:,:] = self.resize_image(img_blur[:,:,0])
                            inputs[i,1,:,:] = self.resize_image(img_blur[:,:,1])
                            inputs[i,2,:,:] = self.resize_image(img_blur[:,:,2])
                targets[i] = 0
                if self.augmented is True:
                    if self.img_x1-self.img_shift >=0.0:
                        tmp_shift = self.img_shift
                    else:
                        tmp_shift = 0.0
                    if self.grayscale is True:
                        inputs[i+1,:,:] = self.resize_image(self.rgb2gray(img_blur[self.img_y1:self.img_y2,self.img_x1-tmp_shift:self.img_x2-tmp_shift]))
                    else:
                        inputs[i+1,0,:,:] = self.resize_image(img_blur[self.img_y1:self.img_y2,self.img_x1-tmp_shift:self.img_x2-tmp_shift,0])
                        inputs[i+1,1,:,:] = self.resize_image(img_blur[self.img_y1:self.img_y2,self.img_x1-tmp_shift:self.img_x2-tmp_shift,1])
                        inputs[i+1,2,:,:] = self.resize_image(img_blur[self.img_y1:self.img_y2,self.img_x1-tmp_shift:self.img_x2-tmp_shift,2])
                    targets[i+1] = 0
                    if self.img_x2+self.img_shift <= 640.0:
                        tmp_shift = self.img_shift
                    else:
                        tmp_shift = 0.0
                    if self.grayscale is True:
                        inputs[i+2,:,:] = self.resize_image(self.rgb2gray(img_blur[self.img_y1:self.img_y2,self.img_x1+tmp_shift:self.img_x2+tmp_shift]))
                    else:
                        inputs[i+2,0,:,:] = self.resize_image(img_blur[self.img_y1:self.img_y2,self.img_x1+tmp_shift:self.img_x2+tmp_shift,0])
                        inputs[i+2,1,:,:] = self.resize_image(img_blur[self.img_y1:self.img_y2,self.img_x1+tmp_shift:self.img_x2+tmp_shift,1])
                        inputs[i+2,2,:,:] = self.resize_image(img_blur[self.img_y1:self.img_y2,self.img_x1+tmp_shift:self.img_x2+tmp_shift,2])                   
                    targets[i+2] = 0
                    if self.img_y2+self.img_shift <= 480.0:
                        tmp_shift = self.img_shift
                    else:
                        tmp_shift = 0.0
                    if self.grayscale is True:
                        inputs[i+3,:,:] = self.resize_image(self.rgb2gray(img_blur[self.img_y1+tmp_shift:self.img_y2+tmp_shift,self.img_x1:self.img_x2]))
                    else:
                        inputs[i+3,0,:,:] = self.resize_image(img_blur[self.img_y1+tmp_shift:self.img_y2+tmp_shift,self.img_x1:self.img_x2,0])
                        inputs[i+3,1,:,:] = self.resize_image(img_blur[self.img_y1+tmp_shift:self.img_y2+tmp_shift,self.img_x1:self.img_x2,1])
                        inputs[i+3,2,:,:] = self.resize_image(img_blur[self.img_y1+tmp_shift:self.img_y2+tmp_shift,self.img_x1:self.img_x2,2])
                    targets[i+3] = 0                
                    i+=4
                else:
                    i+=1

        # extra pics
        if self.num_class > 2:
            k=2
            print("Extra classes detected, reading more pictures")
            for extra_pics_list in extra_pics_listoflist:
                for pic in extra_pics_list:
                    if i < multiplier * (k+1) * self.n_train_max / self.num_class:
                        if self.verbose == True:
                            print("Reading (training) extra class:"+str(k)+", pic "+str(i)+":"+pic)
                        img = misc.imread(pic)
                        if self.grayscale is True:
                            if img.shape == (self.img_width, self.img_height):
                                inputs[i,:,:] = img[:,:]
                            else:
				if blur is True:
	                                img_blur = ndimage.gaussian_filter(img,sigma=0.1)
				else:
					img_blur = img
                                if self.zoom_mode is True:
	                                inputs[i,:,:] = self.resize_image(self.rgb2gray(img_blur[self.img_y1:self.img_y2,self.img_x1:self.img_x2]))
				else:
	                                inputs[i,:,:] = self.resize_image(self.rgb2gray(img_blur))
                        else:
                            img_blur = img
                            if self.zoom_mode is True:
                            	inputs[i,0,:,:] = self.resize_image(img_blur[self.img_y1:self.img_y2,self.img_x1:self.img_x2,0])
                            	inputs[i,1,:,:] = self.resize_image(img_blur[self.img_y1:self.img_y2,self.img_x1:self.img_x2,1])
                            	inputs[i,2,:,:] = self.resize_image(img_blur[self.img_y1:self.img_y2,self.img_x1:self.img_x2,2])
                    	    else:
                            	inputs[i,0,:,:] = self.resize_image(img_blur[:,:,0])
                            	inputs[i,1,:,:] = self.resize_image(img_blur[:,:,1])
                            	inputs[i,2,:,:] = self.resize_image(img_blur[:,:,2])

                        targets[i] = k
                        if self.augmented is True:
                        #if True is True:
                            if self.img_x1-self.img_shift >=0.0:
                                tmp_shift = self.img_shift
                            else:
                                tmp_shift = 0.0
                            if self.grayscale is True:
                                inputs[i+1,:,:] = self.resize_image(self.rgb2gray(img_blur[self.img_y1:self.img_y2,self.img_x1-tmp_shift:self.img_x2-tmp_shift]))
                            else:
                                inputs[i+1,0,:,:] = self.resize_image(img_blur[self.img_y1:self.img_y2,self.img_x1-tmp_shift:self.img_x2-tmp_shift,0])
                                inputs[i+1,1,:,:] = self.resize_image(img_blur[self.img_y1:self.img_y2,self.img_x1-tmp_shift:self.img_x2-tmp_shift,1])
                                inputs[i+1,2,:,:] = self.resize_image(img_blur[self.img_y1:self.img_y2,self.img_x1-tmp_shift:self.img_x2-tmp_shift,2])
                            targets[i+1] = k
                            if self.img_x2+self.img_shift <= 640.0:
                                tmp_shift = self.img_shift
                            else:
                                tmp_shift = 0.0
                            if self.grayscale is True:
                                inputs[i+2,:,:] = self.resize_image(self.rgb2gray(img_blur[self.img_y1:self.img_y2,self.img_x1+tmp_shift:self.img_x2+tmp_shift]))
                            else:
                                inputs[i+2,0,:,:] = self.resize_image(img_blur[self.img_y1:self.img_y2,self.img_x1+tmp_shift:self.img_x2+tmp_shift,0])
                                inputs[i+2,1,:,:] = self.resize_image(img_blur[self.img_y1:self.img_y2,self.img_x1+tmp_shift:self.img_x2+tmp_shift,1])
                                inputs[i+2,2,:,:] = self.resize_image(img_blur[self.img_y1:self.img_y2,self.img_x1+tmp_shift:self.img_x2+tmp_shift,2])                   
                            targets[i+2] = k
                            if self.img_y2+self.img_shift <= 480.0:
                                tmp_shift = self.img_shift
                            else:
                                tmp_shift = 0.0
                            if self.grayscale is True:
                                inputs[i+3,:,:] = self.resize_image(self.rgb2gray(img_blur[self.img_y1+tmp_shift:self.img_y2+tmp_shift,self.img_x1:self.img_x2]))
                            else:
                                inputs[i+3,0,:,:] = self.resize_image(img_blur[self.img_y1+tmp_shift:self.img_y2+tmp_shift,self.img_x1:self.img_x2,0])
                                inputs[i+3,1,:,:] = self.resize_image(img_blur[self.img_y1+tmp_shift:self.img_y2+tmp_shift,self.img_x1:self.img_x2,1])
                                inputs[i+3,2,:,:] = self.resize_image(img_blur[self.img_y1+tmp_shift:self.img_y2+tmp_shift,self.img_x1:self.img_x2,2])
                            targets[i+3] = k                
                            i+=4
                        else:
                            i+=1
                k+=1

	if self.pretrain_model is True:
		# requirements for pretraining
		# change from RGB to BGR (misc.imread is RGB)
		if self.pretrain_name is not "inception":
			inputs = inputs[:,::-1,:,:]
			if self.grayscale is False:
				inputs = inputs - self.IMAGE_MEAN
			else:
				inputs = inputs - self.rgb2gray(self.IMAGE_MEAN)
		else:
			# inceptions uses RGB
			# scale to -1, 1
			inputs = (inputs - 128.0000) / 128.0000
	else:
	        inputs = (inputs / 255.0000) - 0.5
        self.X_train = inputs
        self.Y_train = targets
        # save the values too
        np.save(close_dir+"/train_data_inputs.cache",inputs)
        np.save(close_dir+"/train_data_targets.cache",targets)
        print("Info: Saved X_train model to "+close_dir+"/train_data_inputs.cache")
        print("Info: Saved Y_train model to "+close_dir+"/train_data_targets.cache")

    # read_test_dir
    def read_test_dir(self,test_dir,blur=True):
        live_gate_pics = list(glob.iglob(test_dir+"/*.jpg"))
        nsamples = len(live_gate_pics)
        test_data  = np.zeros(shape=(nsamples,self.img_height,self.img_width)) # only at the section of interest
        i=0
        for pic in live_gate_pics:
                if self.verbose == True:
                    print("Reading (test) pic " + str(i) + ":"+pic)
                img = misc.imread(pic)
		if blur is True:
	                img_blur = ndimage.gaussian_filter(img,sigma=0.1)
		else:
			img_blur = img
                img_gray = self.resize_image(self.rgb2gray(img_blur[self.img_y1:self.img_y2,self.img_x1:self.img_x2]))
                test_data[i,:,:] = img_gray
                i+=1
        test_data = (test_data / 255.0000) - 0.5
        self.X_test = test_data

    # training - read all and save to npy files
    def read_train_dirs_all(self,pics_dir=list(),all_class_represented=True,output_dir="./",type="jpg",blur=True):
        # for multi-class support
        pics_dir_len = 0
        pics_dir_pics_listoflist = [list()] * self.num_class
        if self.num_class > 0:
            for i in range(0,self.num_class):
                print("Class:"+pics_dir[i])
                pics_dir_pics_listoflist[i].append(list(glob.iglob(pics_dir[i]+"/*."+type)))
                pics_dir_len += len(list(glob.iglob(pics_dir[i]+"/*."+type)))
        multiplier = 1
        total_samples = pics_dir_len
        nsamples = min(min(pics_dir_len,ceil((self.num_class) * self.n_train_max/self.num_class)),self.n_train_max)
        print("Info: Number of samples in current chunk:"+str(nsamples))            
        if self.grayscale is True:
            inputs  = np.zeros(shape=(nsamples*multiplier,self.img_height,self.img_width)) # only at the section of interest
        else:
            inputs  = np.zeros(shape=(nsamples*multiplier,3,self.img_height,self.img_width)) # RGB
        targets = np.zeros(shape=(nsamples*multiplier,)) # vector only
        i=0
        chunk_cnt = 0
        accumulated_samples = 0
        dont_save_next = 0 
        while nsamples != 0:
            new_pics_dir_len = 0
            new_pics_dir_listoflist = [list()] * self.num_class
            if self.num_class > 0:
                k=0
                for pics_dir_pics_list in pics_dir_pics_listoflist[k]:
                    class_pics_list = list()
                    for pic in pics_dir_pics_list:
                        if i < multiplier * (k+1) * self.n_train_max / self.num_class:
                            if self.verbose == True:
                                print("Info: Chunk "+str(chunk_cnt)+": Reading (training) class:"+str(k)+", pic "+str(i)+":"+str(pic))
                            img = misc.imread(pic)
                            if self.grayscale is True:
                                if img.shape == (self.img_width, self.img_height):
                                    inputs[i,:,:] = img[:,:]
                                else:
				    if blur is True:
	                                img_blur = ndimage.gaussian_filter(img,sigma=0.1)
				    else:
					img_blur = img    
                                    if self.zoom_mode is True:
                                        inputs[i,:,:] = self.resize_image(self.rgb2gray(img_blur[self.img_y1:self.img_y2,self.img_x1:self.img_x2]))
                                    else:
	                                inputs[i,:,:] = self.resize_image(self.rgb2gray(img_blur))
                            else:
                                img_blur = img
                                if self.zoom_mode is True:
                                    inputs[i,0,:,:] = self.resize_image(img_blur[self.img_y1:self.img_y2,self.img_x1:self.img_x2,0])
                                    inputs[i,1,:,:] = self.resize_image(img_blur[self.img_y1:self.img_y2,self.img_x1:self.img_x2,1])
                                    inputs[i,2,:,:] = self.resize_image(img_blur[self.img_y1:self.img_y2,self.img_x1:self.img_x2,2])
                                else:
                                    inputs[i,0,:,:] = self.resize_image(img_blur[:,:,0])
                                    inputs[i,1,:,:] = self.resize_image(img_blur[:,:,1])
                                    inputs[i,2,:,:] = self.resize_image(img_blur[:,:,2])
                            targets[i] = k
                            i+=1 # inc index per class
                            accumulated_samples += 1
                        else:
                            # save for next iteration since it was not read this time
                            class_pics_list.append(pic)
                    if len(class_pics_list) > 0:
                        new_pics_dir_listoflist[k].append(class_pics_list)
                        new_pics_dir_len += len(class_pics_list)
                    else:
                        if all_class_represented is True:
                            dont_save_next = 1
                            #print("INFO: Skip saving next round!")
                        new_pics_dir_listoflist[k].append(list())
                    k+=1 # inc class
                    
            if self.pretrain_model is True:
		# requirements for pretraining
		# change from RGB to BGR (misc.imread is RGB)
		if self.pretrain_name is not "inception":
			inputs = inputs[:,::-1,:,:]
			if self.grayscale is False:
				inputs = inputs - self.IMAGE_MEAN
			else:
				inputs = inputs - self.rgb2gray(self.IMAGE_MEAN)
		else:
			# inceptions uses RGB
			# scale to -1, 1
			inputs = (inputs - 128.0000) / 128.0000
            else:
	        inputs = (inputs / 255.0000) - 0.5


            self.X_train = inputs
            self.Y_train = targets
            # save the values too
            np.save(output_dir+"train_data_inputs."+str(chunk_cnt)+".cache",inputs)
            np.save(output_dir+"train_data_targets."+str(chunk_cnt)+".cache",targets)
            print("Info: Saved X_train model to "+output_dir+"train_data_inputs."+str(chunk_cnt)+".cache")
            print("Info: Saved Y_train model to "+output_dir+"train_data_targets."+str(chunk_cnt)+".cache")
                
            # recalculate nsamples and reassign list
            pics_dir_len = new_pics_dir_len
            pics_dir_pics_listoflist = new_pics_dir_listoflist
            if i != nsamples:
                print("ERROR: Iteration does not match number of samples! Will cause issues if this is not fixed...")
                sys.exit(999)

            if dont_save_next == 1:
                break
            # recalculate numpy array size
            nsamples=0
            i=0
            k=0
            for pics_dir_pics_list in pics_dir_pics_listoflist[k]:
                for pic in pics_dir_pics_list:
                    if i < multiplier * (k+1) * self.n_train_max / self.num_class:
                        nsamples += 1
                        i+=1
                k+=1
            print("Info: Number of samples in current chunk:"+str(nsamples))            
            if self.grayscale is True:
                inputs  = np.zeros(shape=(nsamples*multiplier,self.img_height,self.img_width)) # only at the section of interest
            else:
                inputs  = np.zeros(shape=(nsamples*multiplier,3,self.img_height,self.img_width)) # RGB
            targets = np.zeros(shape=(nsamples*multiplier,)) # vector only
            i=0
            chunk_cnt += 1 # increment chunk count           
        if accumulated_samples == total_samples:
            print("OK: Total samples = "+str(total_samples)+", Accumulated samples = "+str(accumulated_samples))
        else:
            print("WARNING: Total samples = "+str(total_samples)+", Accumulated samples = "+str(accumulated_samples)+" - mismatch noted!")
            
    # training - read all and save to npy files
    # masks are always binary, so skip color
    def read_mask_dirs_all(self,pics_dir=list(),masks_dir=list(),all_class_represented=True,output_dir="./",type="jpg"):
        # for multi-class support
        pics_dir_len = 0
        pics_dir_pics_listoflist = [list()] * self.num_class
        if self.num_class > 0:
            for i in range(0,self.num_class):
                print("Class:"+pics_dir[i])
                pics_dir_pics_listoflist[i].append(list(glob.iglob(pics_dir[i]+"/*."+type)))
                pics_dir_len += len(list(glob.iglob(pics_dir[i]+"/*."+type)))
        multiplier = 1
        total_samples = pics_dir_len
        nsamples = min(min(pics_dir_len,ceil((self.num_class) * self.n_train_max/self.num_class)),self.n_train_max)
        print("Info: Number of samples in current chunk:"+str(nsamples))            
        targets  = np.zeros(shape=(nsamples*multiplier,self.img_height,self.img_width)) # only at the section of interest
        i=0
        chunk_cnt = 0
        accumulated_samples = 0
        dont_save_next = 0 
        while nsamples != 0:
            new_pics_dir_len = 0
            new_pics_dir_listoflist = [list()] * self.num_class
            if self.num_class > 0:
                k=0
                for pics_dir_pics_list in pics_dir_pics_listoflist[k]:
                    class_pics_list = list()
                    for pic in pics_dir_pics_list:
			sobj = re.search(r'([^\/\\]+)\.'+type,pic)
			if re.search(r'[\.]',sobj.group(1)):
				mask_name = re.sub(r'[\.]','_mask.',sobj.group(1))				
			else:
				mask_name = re.sub(r'$','_mask',sobj.group(1))
			pic = masks_dir[0] + "/" + mask_name + "." + type

                        if i < multiplier * (k+1) * self.n_train_max / self.num_class:
                            if self.verbose == True:
                                print("Info: Chunk "+str(chunk_cnt)+": Reading (training) target:"+str(k)+", pic "+str(i)+":"+str(pic))
                            img = misc.imread(pic)
                            if img.shape == (self.img_width, self.img_height):
                                targets[i,:,:] = img[:,:]
                            else:
                                img_blur = img
                                if self.zoom_mode is True:
                                    targets[i,:,:] = self.resize_image(self.rgb2gray(img_blur[self.img_y1:self.img_y2,self.img_x1:self.img_x2]))
                                else:
                                    targets[i,:,:] = self.resize_image(self.rgb2gray(img_blur))
                            i+=1 # inc index per class
                            accumulated_samples += 1
                        else:
                            # save for next iteration since it was not read this time
                            class_pics_list.append(pic)
                    if len(class_pics_list) > 0:
                        new_pics_dir_listoflist[k].append(class_pics_list)
                        new_pics_dir_len += len(class_pics_list)
                    else:
                        if all_class_represented is True:
                            dont_save_next = 1
                            #print("INFO: Skip saving next round!")
                        new_pics_dir_listoflist[k].append(list())
                    k+=1 # inc class
                    
            targets = (targets / 255.0000)

            self.Y_train = targets
            # save the values too
            np.save(output_dir+"train_data_targets."+str(chunk_cnt)+".cache",targets)
            print("Info: Saved Y_train model (masks) to "+output_dir+"train_data_targets."+str(chunk_cnt)+".cache")
                
            # recalculate nsamples and reassign list
            pics_dir_len = new_pics_dir_len
            pics_dir_pics_listoflist = new_pics_dir_listoflist
            if i != nsamples:
                print("ERROR: Iteration does not match number of samples! Will cause issues if this is not fixed...")
                sys.exit(999)

            if dont_save_next == 1:
                break
            # recalculate numpy array size
            nsamples=0
            i=0
            k=0
            for pics_dir_pics_list in pics_dir_pics_listoflist[k]:
                for pic in pics_dir_pics_list:
                    if i < multiplier * (k+1) * self.n_train_max / self.num_class:
                        nsamples += 1
                        i+=1
                k+=1
            print("Info: Number of samples in current chunk:"+str(nsamples))            

            targets  = np.zeros(shape=(nsamples*multiplier,self.img_height,self.img_width)) # only at the section of interest
            i=0
            chunk_cnt += 1 # increment chunk count           
        if accumulated_samples == total_samples:
            print("OK: Total samples = "+str(total_samples)+", Accumulated samples = "+str(accumulated_samples))
        else:
            print("WARNING: Total samples = "+str(total_samples)+", Accumulated samples = "+str(accumulated_samples)+" - mismatch noted!")

    # load your training dataset
    def load_train(self,X_pkl,Y_pkl):
        self.X_train = np.load(X_pkl)
        self.Y_train = np.load(Y_pkl)
        print("Info: Training set loaded to X_train and Y_train")

    # save your training model
    def save_model(self,pkl):
        with open(pkl + ".pkl", 'wb') as f:
            cPickle.dump(self.classifier, f, -1)
        self.classifier.save_params_to(pkl)
        print("Info: Saved CNN params to "+pkl)

    # load your model
    def load_model(self, pkl):
        self.classifier.load_params_from(pkl)
        print("Info: Loaded params from "+pkl)

    # load external model
    def load_external_model(self, pkl,with_params=True):
	with open(pkl, 'rb') as f:
		params = pickle.load(f)
	self.classifier.initialize_layers()
	if with_params:
		set_all_param_values(self.classifier.layers_.values(), params['param values'])
	else:
		set_all_param_values(self.classifier.layers_.values(), params['values'])

    # shows image for one file (post-grayscale)
    def show_image(self,file,grayscale=True):
        image = misc.imread(file)
        if grayscale is True:
            img_gray = self.rgb2gray(image)
            plt.imshow(img_gray[self.img_y1:self.img_y2,self.img_x1:self.img_x2], cmap=plt.cm.Greys_r)
        else:
            plt.imshow(image)
        plt.show()

    def show_image_dataset(self,X):
        if self.grayscale is True:
            plt.imshow(X, cmap=plt.cm.Greys_r)
        else:
            # reshape this so that RGB is not last index in matrix
            X_reshape = np.zeros(shape=(np.size(X,1),np.size(X,2),3))
            X_reshape[:,:,0] = X[0,:,:]
            X_reshape[:,:,1] = X[1,:,:]
            X_reshape[:,:,2] = X[2,:,:]

	    if self.pretrain_model is False:
	            plt.imshow(uint8(255.000*(X_reshape+0.5)))
	    else:
		    if self.pretrain_name is not "inception":
			    plt.imshow(uint8((X_reshape+self.IMAGE_MEAN.reshape(1,1,3))[:,:,::-1]))
		    else:
			    plt.imshow(uint8((X_reshape*128.000)+128.0000))
        plt.show()

    # manual grayscale conversion
    def rgb2gray(self,rgb):
	# already grayscale, skip this
	if len(shape(rgb)) == 2:
		return rgb
	else:
	        return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

    # fit CNN
    def fit(self,X,Y):
        self.classifier.fit(X,Y)

    # predict jpg
    def predict(self,X):
        prediction = self.classifier.predict(X)
        return prediction

    # standard MLP, one hidden layer
    def define_mlp_type1(self):
        self.classifier = NeuralNet(
            layers = [
                ('input', layers.InputLayer),
                ('hidden1', layers.DenseLayer),
                ('dropout1', layers.DropoutLayer),
                ('output', layers.DenseLayer)
                ],
            input_shape = (None, self.img_ch, self.img_height, self.img_width),
            dropout1_p=0.50,
            hidden1_num_units = 2000,
            output_num_units = self.num_class,
            output_nonlinearity = None,

            #optimization parameters:
            update = nesterov_momentum,
            update_learning_rate = 0.000005,
            update_momentum = 0.9,
            regression = True,
            max_epochs = 2500,
            verbose = 2
        )
        self.classifier.initialize()
        layers_info = PrintLayerInfo()
        layers_info(self.classifier)

    # define basic CNN with 2 conv layers + 2 maxpool
    # this architecture converges to around
    # - train loss = 0.001700
    # - val loss = 0.002330
    # For reference, the 2nd best train/val loss is around 0.001306 and 0.001121 based on dnouri's posts (3000 epochs with dropout)
    # For reference, the best train/val loss is in the order of 0.000760/0.000787 with 10k epochs
    # - ratio = 0.75217 (a bit of overfitting, but not too bad)
    # - training accuracy is 98.64% using (predict(X_train...) == Y_train).mean()
    def define_cnn_type1(self):
        # one possibility
        self.classifier = NeuralNet(
            layers = [
		('input', layers.InputLayer),
		('conv1', layers.Conv2DLayer),
		('pool1', layers.MaxPool2DLayer),
		('conv2', layers.Conv2DLayer),
		('pool2', layers.MaxPool2DLayer),
		("hidden3", layers.DenseLayer),
		("output", layers.DenseLayer),
		],
		#layer parameters:
		input_shape = (None, self.img_ch, self.img_height, self.img_width),
		conv1_num_filters = 16, conv1_filter_size = (7, 7), pool1_pool_size = (2,2),
		conv2_num_filters = 32, conv2_filter_size = (7, 7), pool2_pool_size = (2,2),
		hidden3_num_units = 100,
		output_nonlinearity = softmax,
                #NOTE: REMEMBER THAT 1 and 0 are two classes, not one!!
		output_num_units = self.num_class, 

		#optimization parameters:
		update = nesterov_momentum,
                update_learning_rate = 0.00005,
                update_momentum = 0.9,
		regression = False,
		max_epochs = 100,
		verbose = 2,
                batch_iterator_train=BatchIterator(batch_size=50)
		)
        self.classifier.initialize()
        layers_info = PrintLayerInfo()
        layers_info(self.classifier)

    # same as type1, but does regression instead
    def define_cnn_type1_b(self):
        # one possibility
        self.classifier = NeuralNet(
            layers = [
		('input', layers.InputLayer),
		('conv1', layers.Conv2DLayer),
		('pool1', layers.MaxPool2DLayer),
		('conv2', layers.Conv2DLayer),
		('pool2', layers.MaxPool2DLayer),
		("hidden3", layers.DenseLayer),
                ('dropout1', layers.DropoutLayer),
		("output", layers.DenseLayer),
		],
		#layer parameters:
		input_shape = (None, self.img_ch, self.img_height, self.img_width),
		conv1_num_filters = 16, conv1_filter_size = (7, 7), pool1_pool_size = (2,2),
		conv2_num_filters = 32, conv2_filter_size = (7, 7), pool2_pool_size = (2,2),
                dropout1_p=0.50,
		hidden3_num_units = 100,
		output_nonlinearity = None,
                #NOTE: REMEMBER THAT 1 and 0 are two classes, not one!!
		output_num_units = self.num_class, 

		#optimization parameters:
		update = nesterov_momentum,
                update_learning_rate = 0.00002,
                update_momentum = 0.9,
		regression = True,
		max_epochs = 5000,
		verbose = 2
                #batch_iterator_train=BatchIterator(batch_size=100)
		)
        self.classifier.initialize()
        layers_info = PrintLayerInfo()
        layers_info(self.classifier)

    # same as type1, but does regression instead
    # a good height/width for this is 64x64
    def define_cnn_type1_b2(self):
        # one possibility
        self.classifier = NeuralNet(
            layers = [
		('input', layers.InputLayer),
		('conv1', layers.Conv2DLayer),
		('pool1', layers.MaxPool2DLayer),
                #('dropout1', layers.DropoutLayer),
		('conv2', layers.Conv2DLayer),
		('pool2', layers.MaxPool2DLayer),
                #('dropout2', layers.DropoutLayer),
                ('conv3', layers.Conv2DLayer),
                ('pool3', layers.MaxPool2DLayer),
                #('dropout3', layers.DropoutLayer),
		("hidden4", layers.DenseLayer),
                #('dropout4', layers.DropoutLayer),
		("output", layers.DenseLayer),
		],
		#layer parameters:
		input_shape = (None, self.img_ch, self.img_height, self.img_width),
		conv1_num_filters = 16, conv1_filter_size = (2, 2), pool1_pool_size = (2,2),
                #dropout1_p=0.10,
		conv2_num_filters = 32, conv2_filter_size = (2, 2), pool2_pool_size = (2,2),
                #dropout2_p=0.20,
		conv3_num_filters = 64, conv3_filter_size = (2, 2), pool3_pool_size = (2,2),
                #dropout3_p=0.30,
		hidden4_num_units = 500,
                #dropout4_p=0.50,
		output_nonlinearity = None,
                #NOTE: REMEMBER THAT 1 and 0 are two classes, not one!!
		output_num_units = self.num_class, 

		#optimization parameters:
		update = nesterov_momentum,
                #update_learning_rate = 0.0005,
                update_learning_rate = 0.0005,
                update_momentum = 0.9,
                #update_learning_rate = theano.shared(float32(0.0005)),
                #update_momentum = theano.shared(float32(0.9)),
		regression = True,
		max_epochs = 2500,
		verbose = 2,
                # this code implements variable learning rate (start at 0.0005 ends at 0.00005)
                #on_epoch_finished=[
                #    AdjustVariable('update_learning_rate',start=0.0005, stop=0.00005),
                #    AdjustVariable('update_momentum', start=0.9, stop=0.9),
                #],
                #batch_iterator_train=BatchIterator(batch_size=100)
		)
        self.classifier.initialize()
        layers_info = PrintLayerInfo()
        layers_info(self.classifier)

    # same as type1, but does regression instead
    # a good height/width for this is 64x64
    def define_cnn_type1_b2_dropout(self):
        # one possibility
        self.classifier = NeuralNet(
            layers = [
		('input', layers.InputLayer),
		('conv1', layers.Conv2DLayer),
		('pool1', layers.MaxPool2DLayer),
                #('dropout1', layers.DropoutLayer),
		('conv2', layers.Conv2DLayer),
		('pool2', layers.MaxPool2DLayer),
                #('dropout2', layers.DropoutLayer),
                ('conv3', layers.Conv2DLayer),
                ('pool3', layers.MaxPool2DLayer),
                #('dropout3', layers.DropoutLayer),
		("hidden4", layers.DenseLayer),
                ('dropout1', layers.DropoutLayer),
		("output", layers.DenseLayer),
		],
		#layer parameters:
		input_shape = (None, self.img_ch, self.img_height, self.img_width),
		conv1_num_filters = 16, conv1_filter_size = (2, 2), pool1_pool_size = (2,2),
                #dropout1_p=0.10,
		conv2_num_filters = 32, conv2_filter_size = (2, 2), pool2_pool_size = (2,2),
                #dropout2_p=0.20,
		conv3_num_filters = 64, conv3_filter_size = (2, 2), pool3_pool_size = (2,2),
                #dropout3_p=0.30,
		hidden4_num_units = 500,
                dropout1_p=0.50,
		output_nonlinearity = None,
                #NOTE: REMEMBER THAT 1 and 0 are two classes, not one!!
		output_num_units = self.num_class, 

		#optimization parameters:
		update = nesterov_momentum,
                #update_learning_rate = 0.0005,
                update_learning_rate = 0.0005,
                update_momentum = 0.9,
                #update_learning_rate = theano.shared(float32(0.0005)),
                #update_momentum = theano.shared(float32(0.9)),
		regression = True,
		max_epochs = 2500,
		verbose = 2,
                # this code implements variable learning rate (start at 0.0005 ends at 0.00005)
                #on_epoch_finished=[
                #    AdjustVariable('update_learning_rate',start=0.0005, stop=0.00005),
                #    AdjustVariable('update_momentum', start=0.9, stop=0.9),
                #],
                #batch_iterator_train=BatchIterator(batch_size=100)
		)
        self.classifier.initialize()
        layers_info = PrintLayerInfo()
        layers_info(self.classifier)

    def define_cnn_type1_b3(self):
        # one possibility
        self.classifier = NeuralNet(
            layers = [
		('input', layers.InputLayer),
		('conv1', layers.Conv2DLayer),
		('pool1', layers.MaxPool2DLayer),
		('conv2', layers.Conv2DLayer),
		('pool2', layers.MaxPool2DLayer),
                ('conv3', layers.Conv2DLayer),
                ('pool3', layers.MaxPool2DLayer),
                ('conv4', layers.Conv2DLayer),
                ('pool4', layers.MaxPool2DLayer),
		("hidden5", layers.DenseLayer),
		("output", layers.DenseLayer),
		],
		#layer parameters:
		input_shape = (None, self.img_ch, self.img_height, self.img_width),
		conv1_num_filters = 32, conv1_filter_size = (5, 5), pool1_pool_size = (2,2),
		conv2_num_filters = 32, conv2_filter_size = (3, 3), pool2_pool_size = (2,2),
		conv3_num_filters = 32, conv3_filter_size = (3, 3), pool3_pool_size = (2,2),
                conv4_num_filters = 32, conv4_filter_size = (3, 3), pool4_pool_size = (2,2),
		hidden5_num_units = 500,
		output_nonlinearity = None,
                #NOTE: REMEMBER THAT 1 and 0 are two classes, not one!!
		output_num_units = self.num_class, 

		#optimization parameters:
		update = nesterov_momentum,
                #update_learning_rate = 0.0005,
                update_learning_rate = 0.0001,
                update_momentum = 0.9,
                #update_learning_rate = theano.shared(float32(0.0005)),
                #update_momentum = theano.shared(float32(0.9)),
		regression = True,
		max_epochs = 10000,
		verbose = 2,
                # this code implements variable learning rate (start at 0.0005 ends at 0.00005)
                #on_epoch_finished=[
                #    AdjustVariable('update_learning_rate',start=0.0005, stop=0.00005),
                #    AdjustVariable('update_momentum', start=0.9, stop=0.9),
                #],
                #batch_iterator_train=BatchIterator(batch_size=100)
		)
        self.classifier.initialize()
        layers_info = PrintLayerInfo()
        layers_info(self.classifier)

    # optimized for facial recognition
    def define_cnn_type1_face(self):
        # one possibility
        self.classifier = NeuralNet(
            layers = [
		('input', layers.InputLayer),
		('conv1', layers.Conv2DLayer),
		('pool1', layers.MaxPool2DLayer),
                #('dropout1', layers.DropoutLayer),
		('conv2', layers.Conv2DLayer),
		('pool2', layers.MaxPool2DLayer),
                #('dropout2', layers.DropoutLayer),
                ('conv3', layers.Conv2DLayer),
                ('pool3', layers.MaxPool2DLayer),
                #('dropout3', layers.DropoutLayer),
		("hidden4", layers.DenseLayer),
                ('dropout1', layers.DropoutLayer),            
		("output", layers.DenseLayer),
		],
		#layer parameters:
		input_shape = (None, self.img_ch, self.img_height, self.img_width),
		conv1_num_filters = 32, conv1_filter_size = (5, 5), pool1_pool_size = (2,2),
                #dropout1_p=0.10, # orig=0.05
		conv2_num_filters = 32, conv2_filter_size = (3, 3), pool2_pool_size = (2,2),
                #dropout2_p=0.40, # orig=0.2
		conv3_num_filters = 32, conv3_filter_size = (3, 3), pool3_pool_size = (2,2),
                #dropout3_p=0.60, # orig=0.4
		hidden4_num_units = 500,
                dropout1_p=0.30, # orig=0.6
		output_nonlinearity = None,
                #NOTE: REMEMBER THAT 1 and 0 are two classes, not one!!
		output_num_units = self.num_class, 

		#optimization parameters:
		update = nesterov_momentum,
                #update_learning_rate = 0.0005,
                update_learning_rate = 0.0004,
                update_momentum = 0.9,
                #update_learning_rate = theano.shared(float32(0.0005)),
                #update_momentum = theano.shared(float32(0.9)),
		regression = True,
		max_epochs = 2500,
		verbose = 2,
                # this code implements variable learning rate (start at 0.0005 ends at 0.00005)
                #on_epoch_finished=[
                #    AdjustVariable('update_learning_rate',start=0.0005, stop=0.00005),
                #    AdjustVariable('update_momentum', start=0.9, stop=0.9),
                #],
                #batch_iterator_train=BatchIterator(batch_size=100)
		)
        self.classifier.initialize()
        layers_info = PrintLayerInfo()
        layers_info(self.classifier)

    # optimized for statefarm driver recognition
    def define_cnn_type1_driver_deep_4lyrs(self):
        # one possibility
        self.classifier = NeuralNet(
            layers = [
                ('input', layers.InputLayer),
                ('conv1', layers.Conv2DLayer),
                ('pool1', layers.MaxPool2DLayer),
                ('conv2', layers.Conv2DLayer),
                ('pool2', layers.MaxPool2DLayer),
                ('conv3', layers.Conv2DLayer),
                ('pool3', layers.MaxPool2DLayer),
                ('conv4', layers.Conv2DLayer),
                ('pool4', layers.MaxPool2DLayer),
                ("hidden5", layers.DenseLayer),
                ('dropout1', layers.DropoutLayer),
                ("output", layers.DenseLayer),
                ],
                input_shape = (None, self.img_ch, self.img_height, self.img_width),
                conv1_num_filters = 32, conv1_filter_size = (5, 5), pool1_pool_size = (2,2),
                conv2_num_filters = 32, conv2_filter_size = (3, 3), pool2_pool_size = (2,2),
                conv3_num_filters = 32, conv3_filter_size = (3, 3), pool3_pool_size = (2,2),
                conv4_num_filters = 32, conv4_filter_size = (3, 3), pool4_pool_size = (2,2),
                hidden5_num_units = 500,
                dropout1_p=0.50, # orig=0.6
                output_nonlinearity = None,
                #NOTE: REMEMBER THAT 1 and 0 are two classes, not one!!
                output_num_units = self.num_class,

                #optimization parameters:
                update = nesterov_momentum,
                update_learning_rate = theano.shared(float32(0.01)),
                update_momentum = theano.shared(float32(0.9)),
                regression = True,
                max_epochs = 1000,
                verbose = 2,
                # this code implements variable learning rate (start at 0.0005 ends at 0.00005)
                on_epoch_finished=[
                    AdjustVariable('update_learning_rate',start=0.01, stop=0.0001),
                    AdjustVariable('update_momentum', start=0.9, stop=0.9),
                ],
                )

    # VGG_CNN_S architecture
    def define_cnn_type1_vgg_cnn_s(self):
	self.classifier = NeuralNet(
            layers = [
                ('input',  layers.InputLayer),
                ('conv1',  layers.dnn.Conv2DDNNLayer),
		('norm1',  layers.LocalResponseNormalization2DLayer),
                ('pool1',  layers.MaxPool2DLayer),
                ('conv2',  layers.dnn.Conv2DDNNLayer),
                ('pool2',  layers.MaxPool2DLayer),
                ('conv3',  layers.dnn.Conv2DDNNLayer),
                ('conv4',  layers.dnn.Conv2DDNNLayer),
                ('conv5',  layers.dnn.Conv2DDNNLayer),
                ('pool3',  layers.MaxPool2DLayer),
                ("hidden6", layers.DenseLayer),
                ('dropout1', layers.DropoutLayer),
                ("hidden7", layers.DenseLayer),
                ('dropout2', layers.DropoutLayer),
                ("output", layers.DenseLayer),
                ],
                #layer parameters:
                input_shape = (None, self.img_ch, self.img_height, self.img_width),
		conv1_num_filters = 96,  conv1_filter_size = (7, 7), conv1_stride = 2, conv1_flip_filters=False,  
		norm1_alpha=0.0001, pool1_pool_size = (3,3), pool1_stride = 3, pool1_ignore_border=False,
		conv2_num_filters = 256, conv2_filter_size = (5, 5), conv2_flip_filters=False, 
		pool2_pool_size = (2,2), pool2_stride = 2, pool2_ignore_border=False,
		conv3_num_filters = 512, conv3_filter_size = (3, 3), conv3_pad = 1, conv3_flip_filters=False,  
		conv4_num_filters = 512, conv4_filter_size = (3, 3), conv4_pad = 1, conv4_flip_filters=False, 
		conv5_num_filters = 512, conv5_filter_size = (3, 3), conv5_pad = 1, conv5_flip_filters=False, 
		pool3_pool_size = (3,3), pool3_stride=3, pool3_ignore_border=False,
                hidden6_num_units = 4096,
                dropout1_p=0.50,
                hidden7_num_units = 4096,
                dropout2_p=0.50,
                output_nonlinearity = None,
                #NOTE: REMEMBER THAT 1 and 0 are two classes, not one!!
                output_num_units = self.num_class,

                #optimization parameters:
                #update = adadelta,
                update = nesterov_momentum,
		# started at 0.005
                update_learning_rate = theano.shared(float32(0.01)),
                update_momentum = theano.shared(float32(0.9)),
                regression = True,
                max_epochs = 250,
                verbose = 3,
                on_epoch_finished=[
                    AdjustVariable('update_learning_rate',start=0.01, stop=0.005),
                    AdjustVariable('update_momentum', start=0.9, stop=0.9),
                ],
                #batch_iterator_train=AllImageIterator(batch_size=16,
                #                                      cache_dir="./kaggle/statefarm/vgg_cache/",
                #                                      augment=False)
		batch_iterator_train=BatchIterator(batch_size=16)
                )
        self.classifier.initialize()
        layers_info = PrintLayerInfo()
        layers_info(self.classifier)

    # Original VGG19
    # works with only 224x224
    def define_cnn_type1_vgg19(self):
        # one possibility
        self.classifier = NeuralNet(
            layers = [
                ('input',  layers.InputLayer),
                ('conv1',  layers.dnn.Conv2DDNNLayer),
                ('conv2',  layers.dnn.Conv2DDNNLayer),
                ('pool1',  layers.MaxPool2DLayer),
                ('conv3',  layers.dnn.Conv2DDNNLayer),
                ('conv4',  layers.dnn.Conv2DDNNLayer),
                ('pool2',  layers.MaxPool2DLayer),
                ('conv5',  layers.dnn.Conv2DDNNLayer),
                ('conv6',  layers.dnn.Conv2DDNNLayer),
                ('conv7',  layers.dnn.Conv2DDNNLayer),
                ('conv8',  layers.dnn.Conv2DDNNLayer),
                ('pool3',  layers.MaxPool2DLayer),
                ('conv9',  layers.dnn.Conv2DDNNLayer),
                ('conv10', layers.dnn.Conv2DDNNLayer),
                ('conv11', layers.dnn.Conv2DDNNLayer),
                ('conv12', layers.dnn.Conv2DDNNLayer),
                ('pool4',  layers.MaxPool2DLayer),
                ('conv13', layers.dnn.Conv2DDNNLayer),
                ('conv14', layers.dnn.Conv2DDNNLayer),
                ('conv15', layers.dnn.Conv2DDNNLayer),
                ('conv16', layers.dnn.Conv2DDNNLayer),
                ('pool5',  layers.MaxPool2DLayer),
                ("hidden17", layers.DenseLayer),
                ('dropout1', layers.DropoutLayer),
                ("hidden18", layers.DenseLayer),
                ('dropout2', layers.DropoutLayer),
                ("output", layers.DenseLayer),
                ],
                #layer parameters:
                input_shape = (None, self.img_ch, self.img_height, self.img_width),
                conv1_num_filters = 64,  conv1_filter_size = (3, 3), conv1_pad = 'same', conv1_flip_filters=False, 
		conv2_num_filters = 64,  conv2_filter_size = (3, 3), conv2_pad = 'same', conv2_flip_filters=False, 
		pool1_pool_size = (2,2), pool1_stride = 2,
                conv3_num_filters = 128, conv3_filter_size = (3, 3), conv3_pad = 'same', conv3_flip_filters=False, 
		conv4_num_filters = 128, conv4_filter_size = (3, 3), conv4_pad = 'same', conv4_flip_filters=False, 
		pool2_pool_size = (2,2), pool2_stride = 2,
                conv5_num_filters = 256, conv5_filter_size = (3, 3), conv5_pad = 'same', conv5_flip_filters=False, 
		conv6_num_filters = 256, conv6_filter_size = (3, 3), conv6_pad = 'same', conv6_flip_filters=False, 
		conv7_num_filters = 256, conv7_filter_size = (3, 3), conv7_pad = 'same', conv7_flip_filters=False, 
                conv8_num_filters = 256, conv8_filter_size = (3, 3), conv8_pad = 'same', conv8_flip_filters=False, 
		pool3_pool_size = (2,2), pool3_stride = 2,
		conv9_num_filters  = 512, conv9_filter_size  = (3, 3), conv9_pad  = 'same', conv9_flip_filters=False, 
		conv10_num_filters = 512, conv10_filter_size = (3, 3), conv10_pad = 'same', conv10_flip_filters=False, 
                conv11_num_filters = 512, conv11_filter_size = (3, 3), conv11_pad = 'same', conv11_flip_filters=False, 
		conv12_num_filters = 512, conv12_filter_size = (3, 3), conv12_pad = 'same', conv12_flip_filters=False, 
		pool4_pool_size = (2,2), pool4_stride = 2,
		conv13_num_filters = 512, conv13_filter_size = (3, 3), conv13_pad = 'same', conv13_flip_filters=False, 
		conv14_num_filters = 512, conv14_filter_size = (3, 3), conv14_pad = 'same', conv14_flip_filters=False, 
		conv15_num_filters = 512, conv15_filter_size = (3, 3), conv15_pad = 'same', conv15_flip_filters=False, 
		conv16_num_filters = 512, conv16_filter_size = (3, 3), conv16_pad = 'same', conv16_flip_filters=False, 
		pool5_pool_size = (2,2), pool5_stride = 2,
                hidden17_num_units = 4096,
                dropout1_p=0.50,
                hidden18_num_units = 4096,
                dropout2_p=0.50,
                output_nonlinearity = None,
                #NOTE: REMEMBER THAT 1 and 0 are two classes, not one!!
                output_num_units = self.num_class,

                #optimization parameters:
                # don't use adadelta, will run out of memory
                #update = nesterov_momentum,
                update = sgd,
                # started at 0.005
                update_learning_rate = theano.shared(float32(0.01)),
                #update_momentum = theano.shared(float32(0.9)),
                regression = True,
                max_epochs = 250,
                verbose = 3,
                on_epoch_finished=[
                    AdjustVariable('update_learning_rate',start=0.01, stop=0.005),
                    #AdjustVariable('update_momentum', start=0.9, stop=0.9),
                ],
                #batch_iterator_train=AllImageIterator(batch_size=12,
                #                                      cache_dir="./kaggle/statefarm/vgg_cache/",
                #                                      augment=False)
                batch_iterator_train=BatchIterator(batch_size=12)
                )
        self.classifier.initialize()
        layers_info = PrintLayerInfo()
        layers_info(self.classifier)
		
    # Original VGG16
    # works with only 224x224
    def define_cnn_type1_driver_deep_vggnet(self):
        # one possibility
        self.classifier = NeuralNet(
            layers = [
                ('input',  layers.InputLayer),
                ('conv1',  layers.dnn.Conv2DDNNLayer),
		('conv2',  layers.dnn.Conv2DDNNLayer),
                ('pool1',  layers.MaxPool2DLayer),
                ('conv3',  layers.dnn.Conv2DDNNLayer),
                ('conv4',  layers.dnn.Conv2DDNNLayer),
		('pool2',  layers.MaxPool2DLayer),
                ('conv5',  layers.dnn.Conv2DDNNLayer),
                ('conv6',  layers.dnn.Conv2DDNNLayer),
                ('conv7',  layers.dnn.Conv2DDNNLayer),
		('pool3',  layers.MaxPool2DLayer),
                ('conv8',  layers.dnn.Conv2DDNNLayer),
                ('conv9',  layers.dnn.Conv2DDNNLayer),
                ('conv10', layers.dnn.Conv2DDNNLayer),
		('pool4',  layers.MaxPool2DLayer),
                ('conv11', layers.dnn.Conv2DDNNLayer),
		('conv12', layers.dnn.Conv2DDNNLayer),
		('conv13', layers.dnn.Conv2DDNNLayer),
		('pool5',  layers.MaxPool2DLayer),
                ("hidden14", layers.DenseLayer),
                ('dropout1', layers.DropoutLayer),
                ("hidden15", layers.DenseLayer),
                ('dropout2', layers.DropoutLayer),
                ("output", layers.DenseLayer),
                ],
                #layer parameters:
                input_shape = (None, self.img_ch, self.img_height, self.img_width),
                conv1_num_filters = 64,  conv1_filter_size = (3, 3), conv1_pad = 'same', conv1_flip_filters=False, conv2_num_filters = 64,  conv2_filter_size = (3, 3), conv2_pad = 'same', conv2_flip_filters=False, pool1_pool_size = (2,2), pool1_stride = 2,
                conv3_num_filters = 128, conv3_filter_size = (3, 3), conv3_pad = 'same', conv3_flip_filters=False, conv4_num_filters = 128, conv4_filter_size = (3, 3), conv4_pad = 'same', conv4_flip_filters=False, pool2_pool_size = (2,2), pool2_stride = 2,
                conv5_num_filters = 256, conv5_filter_size = (3, 3), conv5_pad = 'same', conv5_flip_filters=False, conv6_num_filters = 256, conv6_filter_size = (3, 3), conv6_pad = 'same', conv6_flip_filters=False, conv7_num_filters = 256, conv7_filter_size = (3, 3), conv7_pad = 'same', conv7_flip_filters=False, pool3_pool_size = (2,2), pool3_stride = 2,
                conv8_num_filters = 512, conv8_filter_size = (3, 3), conv8_pad = 'same', conv8_flip_filters=False, conv9_num_filters = 512, conv9_filter_size = (3, 3), conv9_pad = 'same', conv9_flip_filters=False, conv10_num_filters = 512, conv10_filter_size = (3, 3), conv10_pad = 'same', conv10_flip_filters=False, pool4_pool_size = (2,2), pool4_stride = 2,
                conv11_num_filters = 512, conv11_filter_size = (3, 3), conv11_pad = 'same', conv11_flip_filters=False, conv12_num_filters = 512, conv12_filter_size = (3, 3), conv12_pad = 'same', conv12_flip_filters=False, conv13_num_filters = 512, conv13_filter_size = (3, 3), conv13_pad = 'same', conv13_flip_filters=False, pool5_pool_size = (2,2), pool5_stride = 2,
                hidden14_num_units = 4096,
                dropout1_p=0.50,
                hidden15_num_units = 4096,
                dropout2_p=0.50, 
                output_nonlinearity = None,
                #NOTE: REMEMBER THAT 1 and 0 are two classes, not one!!
                output_num_units = self.num_class,

                #optimization parameters:
		# don't use adadelta, will run out of memory
                #update = nesterov_momentum,
		update = sgd,
                # started at 0.005
                update_learning_rate = theano.shared(float32(0.005)),
                #update_momentum = theano.shared(float32(0.9)),
                regression = True,
                max_epochs = 250,
                verbose = 3,
                on_epoch_finished=[
                    AdjustVariable('update_learning_rate',start=0.005, stop=0.001),
                    #AdjustVariable('update_momentum', start=0.9, stop=0.9),
                ],
                batch_iterator_train=AllImageIterator(batch_size=16,
                                                      cache_dir="./kaggle/statefarm/vgg_cache/",
                                                      augment=False)
		#batch_iterator_train=BatchIterator(batch_size=16)
                )
        self.classifier.initialize()
        layers_info = PrintLayerInfo()
        layers_info(self.classifier)

    # optimized for statefarm driver recognition
    def define_cnn_type1_driver_deep_6lyrs(self):
        # one possibility
        self.classifier = NeuralNet(
            layers = [
                ('input', layers.InputLayer),
                ('conv1', layers.Conv2DLayer),
                ('pool1', layers.MaxPool2DLayer),
                ('conv2', layers.Conv2DLayer),
                ('pool2', layers.MaxPool2DLayer),
                ('conv3', layers.Conv2DLayer),
                ('pool3', layers.MaxPool2DLayer),
                ('conv4', layers.Conv2DLayer),
                ('pool4', layers.MaxPool2DLayer),
                ('conv5', layers.Conv2DLayer),
                ('pool5', layers.MaxPool2DLayer),
                ('conv6', layers.Conv2DLayer),
                ('pool6', layers.MaxPool2DLayer),
                ("hidden7", layers.DenseLayer),
                ('dropout1', layers.DropoutLayer),
                ("hidden8", layers.DenseLayer),
                ('dropout2', layers.DropoutLayer),
                ("output", layers.DenseLayer),
                ],
                #layer parameters:
                input_shape = (None, self.img_ch, self.img_height, self.img_width),
                conv1_num_filters = 32, conv1_filter_size = (5, 5), pool1_pool_size = (2,2),
                conv2_num_filters = 32, conv2_filter_size = (3, 3), pool2_pool_size = (2,2),
                conv3_num_filters = 32, conv3_filter_size = (3, 3), pool3_pool_size = (2,2),
                conv4_num_filters = 32, conv4_filter_size = (3, 3), pool4_pool_size = (2,2),
                conv5_num_filters = 32, conv5_filter_size = (3, 3), pool5_pool_size = (2,2),
                conv6_num_filters = 32, conv6_filter_size = (3, 3), pool6_pool_size = (2,2),
                hidden7_num_units = 500,
                dropout1_p=0.50, # orig=0.6
                hidden8_num_units = 500,
                dropout2_p=0.50, # orig=0.6
                output_nonlinearity = None,
                #NOTE: REMEMBER THAT 1 and 0 are two classes, not one!!
                output_num_units = self.num_class,

                #optimization parameters:
                update = nesterov_momentum,
                update_learning_rate = theano.shared(float32(0.01)),
                update_momentum = theano.shared(float32(0.9)),
                regression = True,
                max_epochs = 2500,
                verbose = 2,
                # this code implements variable learning rate (start at 0.0005 ends at 0.00005)
                on_epoch_finished=[
                    AdjustVariable('update_learning_rate',start=0.01, stop=0.00008),
                    AdjustVariable('update_momentum', start=0.9, stop=0.9),
                ],
                )

    # optimized for statefarm driver recognition
    def define_cnn_type1_driver_vgglite(self):
	# 2cv x 2cv x 3cv x 3cv + 3 hidden = 16 layers
        self.classifier = NeuralNet(
            layers = [
                ('input', layers.InputLayer),
                ('conv1', layers.Conv2DLayer),
                ('conv2', layers.Conv2DLayer),
                ('pool1', layers.MaxPool2DLayer),
                ('conv3', layers.Conv2DLayer),
                ('conv4', layers.Conv2DLayer),
                ('pool2', layers.MaxPool2DLayer),
                ('conv5', layers.Conv2DLayer),
		('conv6', layers.Conv2DLayer),
		('conv7', layers.Conv2DLayer),
                ('pool3', layers.MaxPool2DLayer),
                ('conv8', layers.Conv2DLayer),
                ('conv9', layers.Conv2DLayer),
                ('conv10', layers.Conv2DLayer),
                ('pool4', layers.MaxPool2DLayer),
                ("hidden11", layers.DenseLayer),
                ('dropout1', layers.DropoutLayer),
                ("hidden12", layers.DenseLayer),
                ('dropout2', layers.DropoutLayer),
                ("hidden13", layers.DenseLayer),
                ('dropout3', layers.DropoutLayer),
                ("output", layers.DenseLayer),
                ],
                #layer parameters:
                input_shape = (None, self.img_ch, self.img_height, self.img_width),
                conv1_num_filters = 32, conv1_filter_size = (3, 3), conv2_num_filters = 32, conv2_filter_size = (3, 3), pool1_pool_size = (2,2),
                conv3_num_filters = 64, conv3_filter_size = (3, 3), conv4_num_filters = 64, conv4_filter_size = (3, 3), pool2_pool_size = (2,2),
                conv5_num_filters = 128, conv5_filter_size = (3, 3), conv6_num_filters = 128, conv6_filter_size = (3, 3), conv7_num_filters = 128, conv7_filter_size = (3, 3), pool3_pool_size = (2,2),
                conv8_num_filters = 256, conv8_filter_size = (3, 3), conv9_num_filters = 256, conv9_filter_size = (3, 3), conv10_num_filters = 256, conv10_filter_size = (3, 3), pool4_pool_size = (2,2),
                hidden11_num_units = 1024,
                dropout1_p=0.50,
                hidden12_num_units = 1024,
                dropout2_p=0.50,
                hidden13_num_units = 256,
                dropout3_p=0.50,
                output_nonlinearity = None,
                #NOTE: REMEMBER THAT 1 and 0 are two classes, not one!!
                output_num_units = self.num_class,

                #optimization parameters:
                update = adadelta,
                #update_learning_rate = theano.shared(float32(0.00001)),
                #update_momentum = theano.shared(float32(0.9)),
                regression = True,
                max_epochs = 1000,
                verbose = 3,
                # this code implements variable learning rate (start at 0.0005 ends at 0.00005)
                #on_epoch_finished=[
                #    AdjustVariable('update_learning_rate',start=0.00001, stop=0.000001),
                    #AdjustVariable('update_momentum', start=0.9, stop=0.9),
                #],
                #batch_iterator_train=BatchIterator(batch_size=30)
                )

    def define_cnn_type1_driver_tele_stacked(self):
        # one possibility
        self.classifier = NeuralNet(
            layers = [
                ('input', layers.InputLayer),
                ('conv1', layers.Conv2DLayer),
                ('pool1', layers.MaxPool2DLayer),
                ('conv2', layers.Conv2DLayer),
                ('conv3', layers.Conv2DLayer),
                ('pool2', layers.MaxPool2DLayer),
                ('conv4', layers.Conv2DLayer),
                ('conv5', layers.Conv2DLayer),
                ('pool3', layers.MaxPool2DLayer),
                ('conv6', layers.Conv2DLayer),
                ('conv7', layers.Conv2DLayer),
                ('pool4', layers.MaxPool2DLayer),
                ("hidden8", layers.DenseLayer),
                ('dropout1', layers.DropoutLayer),
                ("hidden9", layers.DenseLayer),
                ('dropout2', layers.DropoutLayer),
                ("output", layers.DenseLayer),
                ],
                #layer parameters:
                input_shape = (None, self.img_ch, self.img_height, self.img_width),
                conv1_num_filters = 64, conv1_filter_size = (3, 3), pool1_pool_size = (2,2),
                conv2_num_filters = 128, conv2_filter_size = (3, 3), conv3_num_filters = 128, conv3_filter_size = (3, 3), pool2_pool_size = (2,2),
                conv4_num_filters = 256, conv4_filter_size = (3, 3), conv5_num_filters = 256, conv5_filter_size = (3, 3), pool3_pool_size = (2,2),
                conv6_num_filters = 512, conv6_filter_size = (3, 3), conv7_num_filters = 512, conv7_filter_size = (3, 3), pool4_pool_size = (2,2),
                hidden8_num_units = 4096,
                dropout1_p=0.50, # orig=0.6
                hidden9_num_units = 512,
                dropout2_p=0.50, # orig=0.6
                output_nonlinearity = None,
                #NOTE: REMEMBER THAT 1 and 0 are two classes, not one!!
                output_num_units = self.num_class,

                #optimization parameters:
                update = nesterov_momentum,
                update_learning_rate = theano.shared(float32(0.05)),
                update_momentum = theano.shared(float32(0.9)),
                regression = True,
                max_epochs = 1000,
                verbose = 2,
                # this code implements variable learning rate (start at 0.0005 ends at 0.00005)
                on_epoch_finished=[
                    AdjustVariable('update_learning_rate',start=0.05, stop=0.001),
                    #AdjustVariable('update_learning_rate',start=0.01, stop=0.005),
                    AdjustVariable('update_momentum', start=0.9, stop=0.9),
                ],
                #batch_iterator_train=BatchIterator(batch_size=64)
                )

    def define_cnn_type1_driver_tele(self):
        # one possibility
        self.classifier = NeuralNet(
            layers = [
                ('input', layers.InputLayer),
                ('conv1', layers.Conv2DLayer),
                ('pool1', layers.MaxPool2DLayer),
                ('conv2', layers.Conv2DLayer),
                ('pool2', layers.MaxPool2DLayer),
                ('conv3', layers.Conv2DLayer),
                ('pool3', layers.MaxPool2DLayer),
                ('conv4', layers.Conv2DLayer),
                ('pool4', layers.MaxPool2DLayer),
                ('conv5', layers.Conv2DLayer),
                ('pool5', layers.MaxPool2DLayer),
                ("hidden6", layers.DenseLayer),
                ('dropout1', layers.DropoutLayer),
                ("hidden7", layers.DenseLayer),
                ('dropout2', layers.DropoutLayer),
                ("output", layers.DenseLayer),
                ],
                #layer parameters:
                input_shape = (None, self.img_ch, self.img_height, self.img_width),
                conv1_num_filters = 64, conv1_filter_size = (5, 5), pool1_pool_size = (2,2),
                conv2_num_filters = 128, conv2_filter_size = (3, 3), pool2_pool_size = (2,2),
                conv3_num_filters = 256, conv3_filter_size = (3, 3), pool3_pool_size = (2,2),
                conv4_num_filters = 512, conv4_filter_size = (3, 3), pool4_pool_size = (2,2),
                conv5_num_filters = 512, conv5_filter_size = (3, 3), pool5_pool_size = (2,2),
                hidden6_num_units = 4096,
                dropout1_p=0.60, # orig=0.6
                hidden7_num_units = 512,
                dropout2_p=0.60, # orig=0.6
                output_nonlinearity = None,
                #NOTE: REMEMBER THAT 1 and 0 are two classes, not one!!
                output_num_units = self.num_class,

                #optimization parameters:
                update = nesterov_momentum,
                update_learning_rate = theano.shared(float32(0.01)),
                update_momentum = theano.shared(float32(0.9)),
                regression = True,
                max_epochs = 1000,
                verbose = 2,
                # this code implements variable learning rate (start at 0.0005 ends at 0.00005)
                on_epoch_finished=[
                    #AdjustVariable('update_learning_rate',start=0.05, stop=0.001),
                    AdjustVariable('update_learning_rate',start=0.01, stop=0.005),		   
                    AdjustVariable('update_momentum', start=0.9, stop=0.9),
                ],
		# use none for 128x128, upper limit ~32 for 256x256, upper limit ~110 for 160x160
		# 64 for 192x192
                batch_iterator_train=BatchIterator(batch_size=20)
                )

    # optimized for statefarm driver recognition
    def define_cnn_type1_driver_deep(self):
        # one possibility
        self.classifier = NeuralNet(
            layers = [
                ('input', layers.InputLayer),
                ('conv1', layers.Conv2DLayer),
                ('pool1', layers.MaxPool2DLayer),
                ('conv2', layers.Conv2DLayer),
                ('pool2', layers.MaxPool2DLayer),
                ('conv3', layers.Conv2DLayer),
                ('pool3', layers.MaxPool2DLayer),
                ('conv4', layers.Conv2DLayer),
                ('pool4', layers.MaxPool2DLayer),
                ('conv5', layers.Conv2DLayer),
                ('pool5', layers.MaxPool2DLayer),
                ("hidden6", layers.DenseLayer),
                ('dropout1', layers.DropoutLayer),
                ("hidden7", layers.DenseLayer),
                ('dropout2', layers.DropoutLayer),
                ("output", layers.DenseLayer),
                ],
                #layer parameters:
                input_shape = (None, self.img_ch, self.img_height, self.img_width),
                conv1_num_filters = 32, conv1_filter_size = (5, 5), pool1_pool_size = (2,2),
                conv2_num_filters = 32, conv2_filter_size = (3, 3), pool2_pool_size = (2,2),
                conv3_num_filters = 32, conv3_filter_size = (3, 3), pool3_pool_size = (2,2),
                conv4_num_filters = 32, conv4_filter_size = (3, 3), pool4_pool_size = (2,2),
                conv5_num_filters = 32, conv5_filter_size = (3, 3), pool5_pool_size = (2,2),
                hidden6_num_units = 500,
                dropout1_p=0.50, # orig=0.6
                hidden7_num_units = 500,
                dropout2_p=0.50, # orig=0.6
                output_nonlinearity = None,
                #NOTE: REMEMBER THAT 1 and 0 are two classes, not one!!
                output_num_units = self.num_class,

                #optimization parameters:
                update = nesterov_momentum,
                update_learning_rate = theano.shared(float32(0.005)),
                update_momentum = theano.shared(float32(0.9)),
                regression = True,
                max_epochs = 1000,
                verbose = 2,
                # this code implements variable learning rate (start at 0.0005 ends at 0.00005)
                on_epoch_finished=[
                    AdjustVariable('update_learning_rate',start=0.005, stop=0.0005),
                    AdjustVariable('update_momentum', start=0.9, stop=0.9),
                ],
#                batch_iterator_train=BatchIterator(batch_size=500)
                )

    # optimized for facial recognition
    def define_cnn_type1_face_long(self):
        # one possibility
        self.classifier = NeuralNet(
            layers = [
		('input', layers.InputLayer),
		('conv1', layers.Conv2DLayer),
		('pool1', layers.MaxPool2DLayer),
                #('dropout1', layers.DropoutLayer),
		('conv2', layers.Conv2DLayer),
		('pool2', layers.MaxPool2DLayer),
                #('dropout2', layers.DropoutLayer),
                ('conv3', layers.Conv2DLayer),
                ('pool3', layers.MaxPool2DLayer),
                #('dropout3', layers.DropoutLayer),
		("hidden4", layers.DenseLayer),
                ('dropout1', layers.DropoutLayer),            
		("output", layers.DenseLayer),
		],
		#layer parameters:
		input_shape = (None, self.img_ch, self.img_height, self.img_width),
		conv1_num_filters = 32, conv1_filter_size = (5, 5), pool1_pool_size = (2,2),
                #dropout1_p=0.10, # orig=0.05
		conv2_num_filters = 32, conv2_filter_size = (3, 3), pool2_pool_size = (2,2),
                #dropout2_p=0.40, # orig=0.2
		conv3_num_filters = 32, conv3_filter_size = (3, 3), pool3_pool_size = (2,2),
                #dropout3_p=0.60, # orig=0.4
		hidden4_num_units = 500,
                dropout1_p=0.50, # orig=0.6
		output_nonlinearity = None,
                #NOTE: REMEMBER THAT 1 and 0 are two classes, not one!!
		output_num_units = self.num_class, 

		#optimization parameters:
		update = nesterov_momentum,
                #update_learning_rate = 0.0005,
                #update_learning_rate = 0.0003,
                #update_momentum = 0.9,
                update_learning_rate = theano.shared(float32(0.0008)),
                update_momentum = theano.shared(float32(0.9)),
		regression = True,
		max_epochs = 1000,
		verbose = 2,
                # this code implements variable learning rate (start at 0.0005 ends at 0.00005)
                on_epoch_finished=[
                    AdjustVariable('update_learning_rate',start=0.0008, stop=0.0003),
                    AdjustVariable('update_momentum', start=0.9, stop=0.9),
                ],
                #batch_iterator_train=BatchIterator(batch_size=100)
		)
        self.classifier.initialize()
        layers_info = PrintLayerInfo()
        layers_info(self.classifier)        

    # VGG16 with dilated convolutions
    def define_cnn_dilated_conv(self):
        # one possibility
        self.classifier = NeuralNet(
            layers = [
                ('input',  layers.InputLayer),
                ('conv1_1',  layers.dnn.Conv2DDNNLayer),
		('conv1_2',  layers.dnn.Conv2DDNNLayer),
                ('pool1',  layers.MaxPool2DLayer),
                ('conv2_1',  layers.dnn.Conv2DDNNLayer),
                ('conv2_2',  layers.dnn.Conv2DDNNLayer),
		('pool2',  layers.MaxPool2DLayer),
                ('conv3_1',  layers.dnn.Conv2DDNNLayer),
                ('conv3_2',  layers.dnn.Conv2DDNNLayer),
                ('conv3_3',  layers.dnn.Conv2DDNNLayer),
		('pool3',  layers.MaxPool2DLayer),
                ('conv4_1',  layers.dnn.Conv2DDNNLayer),
                ('conv4_2',  layers.dnn.Conv2DDNNLayer),
                ('conv4_3', layers.dnn.Conv2DDNNLayer),
		('pool4',  layers.MaxPool2DLayer),
                ('conv5_1', layers.DilatedConv2DLayer),
		('conv5_2', layers.DilatedConv2DLayer),
		('conv5_3', layers.DilatedConv2DLayer),
                ("fc6",     layers.DilatedConv2DLayer),
                ('dropout1', layers.DropoutLayer),
                ("fc7",     layers.dnn.Conv2DDNNLayer),
                ('dropout2', layers.DropoutLayer),
                ("fcfinal", layers.dnn.Conv2DDNNLayer),
                ("ct_conv1_1", layers.DilatedConv2DLayer),
                ("pad_conv1_1", layers.PadLayer),
                ("ct_conv1_2", layers.DilatedConv2DLayer),
                ("ct_conv2_1", layers.DilatedConv2DLayer),
                ("ct_conv3_1", layers.DilatedConv2DLayer),
                ("ct_conv4_1", layers.DilatedConv2DLayer),
                ("ct_conv5_1", layers.DilatedConv2DLayer),
                ("ct_fc1", layers.DilatedConv2DLayer),
                ("ct_fcfinal", layers.DilatedConv2DLayer),
                ],
                #layer parameters:
                input_shape = (None, self.img_ch, self.img_height, self.img_width),
                conv1_1_num_filters = 64,  conv1_1_filter_size = (3, 3), conv1_1_pad = 'same', conv1_1_flip_filters=False, conv1_1_nonlinearity=rectify, conv1_2_num_filters = 64,  conv1_2_filter_size = (3, 3), conv1_2_pad = 'same', conv1_2_flip_filters=False, conv1_2_nonlinearity=rectify, pool1_pool_size = (2,2), pool1_stride = 2,
                conv2_1_num_filters = 128, conv2_1_filter_size = (3, 3), conv2_1_pad = 'same', conv2_1_flip_filters=False, conv2_1_nonlinearity=rectify, conv2_2_num_filters = 128, conv2_2_filter_size = (3, 3), conv2_2_pad = 'same', conv2_2_flip_filters=False, conv2_2_nonlinearity=rectify, pool2_pool_size = (2,2), pool2_stride = 2,
                conv3_1_num_filters = 256, conv3_1_filter_size = (3, 3), conv3_1_pad = 'same', conv3_1_flip_filters=False, conv3_1_nonlinearity=rectify, conv3_2_num_filters = 256, conv3_2_filter_size = (3, 3), conv3_2_pad = 'same', conv3_2_flip_filters=False, conv3_2_nonlinearity=rectify, conv3_3_num_filters = 256, conv3_3_filter_size = (3, 3), conv3_3_pad = 'same', conv3_3_flip_filters=False, conv3_3_nonlinearity=rectify, pool3_pool_size = (2,2), pool3_stride = 2,
                conv4_1_num_filters = 512, conv4_1_filter_size = (3, 3), conv4_1_pad = 'same', conv4_1_flip_filters=False, conv4_1_nonlinearity=rectify, conv4_2_num_filters = 512, conv4_2_filter_size = (3, 3), conv4_2_pad = 'same', conv4_2_flip_filters=False, conv4_2_nonlinearity=rectify, conv4_3_num_filters = 512, conv4_3_filter_size = (3, 3), conv4_3_pad = 'same', conv4_3_flip_filters=False, conv4_3_nonlinearity=rectify, pool4_pool_size = (2,2), pool4_stride = 2,
                conv5_1_num_filters = 512, conv5_1_filter_size = (3, 3), conv5_1_pad = 0, conv5_1_flip_filters=False, conv5_1_dilation = (2,2), conv5_1_nonlinearity=rectify, 
                conv5_2_num_filters = 512, conv5_2_filter_size = (3, 3), conv5_2_pad = 0, conv5_2_flip_filters=False, conv5_2_dilation = (2,2), conv5_2_nonlinearity=rectify, 
                conv5_3_num_filters = 512, conv5_3_filter_size = (3, 3), conv5_3_pad = 0, conv5_3_flip_filters=False, conv5_3_dilation = (2,2), conv5_3_nonlinearity=rectify, 
                fc6_num_filters = 4096, fc6_filter_size = (7,7), fc6_pad = 0, fc6_dilation = (4,4), fc6_nonlinearity=rectify,
                dropout1_p=0.50,
                fc7_num_filters = 4096, fc7_filter_size = (1,1), fc7_pad = 0, fc7_nonlinearity=rectify,
                dropout2_p=0.50, 
                fcfinal_num_filters=self.num_class, fcfinal_filter_size = (1,1), fcfinal_pad = 'same', fcfinal_nonlinearity=None,
                ct_conv1_1_num_filters = 2*self.num_class, ct_conv1_1_filter_size = (3,3), ct_conv1_1_pad = 0, ct_conv1_1_dilation = (1,1), ct_conv1_1_nonlinearity=rectify,
                pad_conv1_1_width = 33,
                ct_conv1_2_num_filters = 2*self.num_class, ct_conv1_2_filter_size = (3,3), ct_conv1_2_pad = 0, ct_conv1_2_dilation = (1,1), ct_conv1_2_nonlinearity=rectify,
                ct_conv2_1_num_filters = 4*self.num_class, ct_conv2_1_filter_size = (3,3), ct_conv2_1_pad = 0, ct_conv2_1_dilation = (2,2), ct_conv2_1_nonlinearity=rectify,
                ct_conv3_1_num_filters = 8*self.num_class, ct_conv3_1_filter_size = (3,3), ct_conv3_1_pad = 0, ct_conv3_1_dilation = (4,4), ct_conv3_1_nonlinearity=rectify,
                ct_conv4_1_num_filters = 16*self.num_class, ct_conv4_1_filter_size = (3,3), ct_conv4_1_pad = 0, ct_conv4_1_dilation = (8,8), ct_conv4_1_nonlinearity=rectify,
                ct_conv5_1_num_filters = 32*self.num_class, ct_conv5_1_filter_size = (3,3), ct_conv5_1_pad = 0, ct_conv5_1_dilation = (16,16), ct_conv5_1_nonlinearity=rectify,
                ct_fc1_num_filters     = 32*self.num_class, ct_fc1_filter_size = (3,3), ct_fc1_pad = 0, ct_fc1_dilation = (1,1), ct_fc1_nonlinearity=rectify,
                ct_final_num_filters   = self.num_class, ct_final_filter_size = (1,1), ct_fcfinal_pad = 0, ct_fcfinal_dilation = (1,1), ct_fcfinal_nonlinearity=softmax,
                #optimization parameters:
		update = adam,
                # started at 0.005
                update_learning_rate = theano.shared(float32(0.00001)),
                regression = True,
                max_epochs = 20,
                verbose = 3,
                on_epoch_finished=[
                    AdjustVariable('update_learning_rate',start=0.00001, stop=0.000005),
                ],
		batch_iterator_train=BatchIterator(batch_size=16)
                )
        self.classifier.initialize()
        layers_info = PrintLayerInfo()
        layers_info(self.classifier)

    # unet
    def define_cnn_unet(self):
        unet = self.build_unet()
        self.classifier = NeuralNet(
                layers = unet,
                #optimization parameters:
                update = adam,
                #update_learning_rate = theano.shared(float32(0.01)),
                update_learning_rate = theano.shared(float32(0.00001)),
                regression = True,
                max_epochs = 2000,
                verbose = 10,
                # this code implements variable learning rate (start at 0.0005 ends at 0.00005)
                on_epoch_finished=[
                    #initial
                    #AdjustVariable('update_learning_rate',start=0.001, stop=0.0005),
                    AdjustVariable('update_learning_rate',start=0.00001, stop=0.000005),
                ],
                #objective_loss_function=self.multilabel_objective,
                objective_loss_function=self.dice_coef,
                batch_iterator_train=BatchIterator(batch_size=1,shuffle=True),
                # needed since normally two dimensions exist for theano - batch, class
                # but now you have an image output target..so...
                y_tensor_type=T.tensor4
                #batch_iterator_train=AllImageIterator(batch_size=128,
                #                                      cache_dir="./kaggle/ultrasound/recomb_cache/",
                #                                      augment=False)
                )
        self.classifier.initialize()
        layers_info = PrintLayerInfo()
        layers_info(self.classifier)

    # recombinator
    def define_cnn_recombinator_deep(self):
        recombinator = self.build_recombinator_network_5lyr()

        self.classifier = NeuralNet(
                layers = recombinator,
                #optimization parameters:
                update = adam,
                update_learning_rate = theano.shared(float32(0.00001)),
                regression = True,
                max_epochs = 2000,
                verbose = 10,
                # this code implements variable learning rate (start at 0.0005 ends at 0.00005)
                on_epoch_finished=[
                    #initial
                    #AdjustVariable('update_learning_rate',start=0.001, stop=0.0005),
                    AdjustVariable('update_learning_rate',start=0.00001, stop=0.000005),
                    #AdjustVariable('update_learning_rate',start=0.01, stop=0.005),
                ],
                #objective_loss_function=self.multilabel_objective,
                objective_loss_function=self.dice_coef,
                batch_iterator_train=BatchIterator(batch_size=12,shuffle=True),
                # needed since normally two dimensions exist for theano - batch, class
                # but now you have an image output target..so...
                y_tensor_type=T.tensor4
                #batch_iterator_train=AllImageIterator(batch_size=128,
                #                                      cache_dir="./kaggle/ultrasound/recomb_cache/",
                #                                      augment=False)
                )
        self.classifier.initialize()
        layers_info = PrintLayerInfo()
        layers_info(self.classifier)

    # recombinator
    def define_cnn_recombinator(self):
	recombinator = self.build_recombinator_network()

        self.classifier = NeuralNet(
                layers = recombinator,
                #optimization parameters:
                update = adam,
                update_learning_rate = theano.shared(float32(0.00001)),
                regression = True,
                max_epochs = 40,
                verbose = 10,
                # this code implements variable learning rate (start at 0.0005 ends at 0.00005)
                on_epoch_finished=[
		    #initial
                    #AdjustVariable('update_learning_rate',start=0.001, stop=0.0005),
		    # for this problem use adam with a low lr 1e-5
                    AdjustVariable('update_learning_rate',start=0.00001, stop=0.000005),
                ],
		#objective_loss_function=self.multilabel_objective,
		objective_loss_function=self.dice_coef,
                batch_iterator_train=BatchIterator(batch_size=12,shuffle=True),
		# needed since normally two dimensions exist for theano - batch, class
		# but now you have an image output target..so...
		y_tensor_type=T.tensor4
                #batch_iterator_train=AllImageIterator(batch_size=128,
                #                                      cache_dir="./kaggle/ultrasound/recomb_cache/",
                #                                      augment=False)
                )
        self.classifier.initialize()
        layers_info = PrintLayerInfo()
        layers_info(self.classifier)	

    # inception v3
    def define_cnn_inception_v3(self):
	# generate all layers using lasagne style, and pass into nolearn wrapper
	inception_v3 = self.build_inception_v3_network()

        self.classifier = NeuralNet(
		layers = inception_v3,
                #optimization parameters:
                update = sgd,
                update_learning_rate = theano.shared(float32(0.05)),
                regression = True,
                max_epochs = 100,
                verbose = 10,
                # this code implements variable learning rate (start at 0.0005 ends at 0.00005)
                on_epoch_finished=[
                    AdjustVariable('update_learning_rate',start=0.05, stop=0.01),
                ],
                #batch_iterator_train=BatchIterator(batch_size=16,shuffle=True)
		batch_iterator_train=AllImageIterator(batch_size=16,
						      cache_dir="./kaggle/statefarm/inception_cache/",
						      augment=False)
                )
        self.classifier.initialize()
        layers_info = PrintLayerInfo()
        layers_info(self.classifier)


    # resnet-152
    def define_cnn_resnet152(self):
        # generate all layers using lasagne style, and pass into nolearn wrapper
        resnet = self.build_resnet50_101_152(layer_cnt=152)

        self.classifier = NeuralNet(
                layers = resnet,
                #optimization parameters:
                update = nesterov_momentum,
                update_learning_rate = theano.shared(float32(0.1)),
                update_momentum = theano.shared(float32(0.9)),
                regression = True,
                max_epochs = 100,
                verbose = 10,
                # this code implements variable learning rate (start at 0.0005 ends at 0.00005)
                on_epoch_finished=[
                    AdjustVariable('update_learning_rate',start=0.1, stop=0.05),
                    AdjustVariable('update_momentum', start=0.9, stop=0.9),
                ],
                batch_iterator_train=BatchIterator(batch_size=16)
                )
        self.classifier.initialize()
        layers_info = PrintLayerInfo()
        layers_info(self.classifier)


    # resnet-101
    def define_cnn_resnet101(self):
        # generate all layers using lasagne style, and pass into nolearn wrapper
        resnet = self.build_resnet50_101_152(layer_cnt=101)

        self.classifier = NeuralNet(
                layers = resnet,
                #optimization parameters:
                update = nesterov_momentum,
                update_learning_rate = theano.shared(float32(0.1)),
                update_momentum = theano.shared(float32(0.9)),
                regression = True,
                max_epochs = 100,
                verbose = 10,
                # this code implements variable learning rate (start at 0.0005 ends at 0.00005)
                on_epoch_finished=[
                    AdjustVariable('update_learning_rate',start=0.1, stop=0.05),
                    AdjustVariable('update_momentum', start=0.9, stop=0.9),
                ],
                batch_iterator_train=BatchIterator(batch_size=16)
                )
        self.classifier.initialize()
        layers_info = PrintLayerInfo()
        layers_info(self.classifier)

    # resnet-50
    def define_cnn_resnet50(self):
        # generate all layers using lasagne style, and pass into nolearn wrapper
        resnet = self.build_resnet50_101_152(layer_cnt=50)

        self.classifier = NeuralNet(
                layers = resnet,
                #optimization parameters:
                update = nesterov_momentum,
                update_learning_rate = theano.shared(float32(0.1)),
                update_momentum = theano.shared(float32(0.9)),
                regression = True,
                max_epochs = 100,
                verbose = 10,
                # this code implements variable learning rate (start at 0.0005 ends at 0.00005)
                on_epoch_finished=[
                    AdjustVariable('update_learning_rate',start=0.1, stop=0.05),
                    AdjustVariable('update_momentum', start=0.9, stop=0.9),
                ],
                batch_iterator_train=BatchIterator(batch_size=16)
                )
        self.classifier.initialize()
        layers_info = PrintLayerInfo()
        layers_info(self.classifier)

    # resnet-34
    def define_cnn_resnet34(self):
        # generate all layers using lasagne style, and pass into nolearn wrapper
        resnet = self.build_resnet34()

        self.classifier = NeuralNet(
                layers = resnet,
                #optimization parameters:
                update = nesterov_momentum,
                update_learning_rate = theano.shared(float32(0.01)),
                update_momentum = theano.shared(float32(0.9)),
                regression = True,
                max_epochs = 100,
                verbose = 10,
                # this code implements variable learning rate (start at 0.0005 ends at 0.00005)
                on_epoch_finished=[
		    # first run = 0.1-0.05, second run = 0.01-0.008
                    AdjustVariable('update_learning_rate',start=0.01, stop=0.005),
                    AdjustVariable('update_momentum', start=0.9, stop=0.9),
                ],
                batch_iterator_train=BatchIterator(batch_size=32,shuffle=True)
                )
        self.classifier.initialize()
        layers_info = PrintLayerInfo()
        layers_info(self.classifier) 

    # same as type1, but does regression instead
    def define_cnn_type1_b_dropout(self):
        # one possibility
        self.classifier = NeuralNet(
            layers = [
		('input', layers.InputLayer),
		('conv1', layers.Conv2DLayer),
		('pool1', layers.MaxPool2DLayer),
                ('dropout1', layers.DropoutLayer),
		('conv2', layers.Conv2DLayer),
		('pool2', layers.MaxPool2DLayer),
                ('dropout2', layers.DropoutLayer),
		("hidden3", layers.DenseLayer),
		("output", layers.DenseLayer),
		],
		#layer parameters:
		input_shape = (None, self.img_ch, self.img_height, self.img_width),
		conv1_num_filters = 16, conv1_filter_size = (7, 7), pool1_pool_size = (2,2),
                dropout1_p=0.50,
		conv2_num_filters = 32, conv2_filter_size = (7, 7), pool2_pool_size = (2,2),
                dropout2_p=0.50,
		hidden3_num_units = 100,
		output_nonlinearity = None,
                #NOTE: REMEMBER THAT 1 and 0 are two classes, not one!!
		output_num_units = self.num_class, 

		#optimization parameters:
		update = nesterov_momentum,
                update_learning_rate = 0.00005,
                update_momentum = 0.9,
		regression = True,
		max_epochs = 500,
		verbose = 2,
                batch_iterator_train=BatchIterator(batch_size=100)
		)
        self.classifier.initialize()
        layers_info = PrintLayerInfo()
        layers_info(self.classifier)

    # same as type1, but does regression instead
    def define_cnn_type1_c(self):
        # one possibility
        self.classifier = NeuralNet(
            layers = [
		('input', layers.InputLayer),
		('conv1', layers.Conv2DLayer),
		('conv2', layers.Conv2DLayer),
                ('conv3', layers.Conv2DLayer),
		('pool1', layers.MaxPool2DLayer),
		('conv4', layers.Conv2DLayer),
		('conv5', layers.Conv2DLayer),
                ('conv6', layers.Conv2DLayer),
		('pool2', layers.MaxPool2DLayer),
		("hidden7", layers.DenseLayer),
		("output", layers.DenseLayer),
		],
		#layer parameters:
		input_shape = (None, self.img_ch, self.img_height, self.img_width),
		conv1_num_filters = 16, conv1_filter_size = (5, 5),
		conv2_num_filters = 16, conv2_filter_size = (3, 3),
                conv3_num_filters = 16, conv3_filter_size = (3, 3),
                pool1_pool_size = (2,2),
                #dropout1_p=0.50,
		conv4_num_filters = 32, conv4_filter_size = (3, 3),
		conv5_num_filters = 32, conv5_filter_size = (3, 3),
                conv6_num_filters = 32, conv6_filter_size = (3, 3),
                pool2_pool_size = (2,2),
                #dropout2_p=0.50,
		hidden7_num_units = 200,
		output_nonlinearity = None,
                #NOTE: REMEMBER THAT 1 and 0 are two classes, not one!!
		output_num_units = self.num_class, 

		#optimization parameters:
		update = nesterov_momentum,
                update_learning_rate = 0.0001,
                update_momentum = 0.9,
		regression = True,
		max_epochs = 500,
		verbose = 2,
                #batch_iterator_train=BatchIterator(batch_size=100)
		)
        self.classifier.initialize()
        layers_info = PrintLayerInfo()
        layers_info(self.classifier)

    # type 2 CNN with dropout and variable learning rate with 2 conv+2 maxpool
    # note: results will only be obvious after 500 epochs or so
    # per epoch ~8s
    def define_cnn_type2(self):
        sys.setrecursionlimit(10000)
        # with dropout
        self.classifier = NeuralNet(
            layers = [
		('input', layers.InputLayer),
		('conv1', layers.Conv2DLayer),
		('pool1', layers.MaxPool2DLayer),
                ('dropout1', layers.DropoutLayer),
		('conv2', layers.Conv2DLayer),
		('pool2', layers.MaxPool2DLayer),
                ('dropout2', layers.DropoutLayer),
		("hidden3", layers.DenseLayer),
		("output", layers.DenseLayer),
		],
		#layer parameters:
		input_shape = (None, self.img_ch, self.img_height, self.img_width),
		conv1_num_filters = 16, conv1_filter_size = (5, 5), pool1_pool_size = (4,4),
                dropout1_p=0.10,
		conv2_num_filters = 32, conv2_filter_size = (5, 5), pool2_pool_size = (4,4),
                dropout2_p=0.10,
		hidden3_num_units = 2048,
		output_nonlinearity = softmax,
                #NOTE: REMEMBER THAT 1 and 0 are two classes, not one!!
		output_num_units = self.num_class, 

		#optimization parameters:
		update = nesterov_momentum,
                # variable learning rate.  come back to this later
                #update_learning_rate = theano.shared(float32(0.0005)),
                #update_momentum = theano.shared(float32(0.9)),
                update_learning_rate = 0.00005,
                update_momentum = 0.9,
		regression = False,
                # this code implements variable learning rate
                #on_epoch_finished=[
                #                    #batch_iterator_train=BatchIterator(batch_size=30)
		#AdjustVariable('update_learning_rate',start=0.0001, stop=0.00001),
                #    AdjustVariable('update_momentum', start=0.9, stop=0.9),
                #],
		max_epochs = 100,
		verbose = 2,
                batch_iterator_train=BatchIterator(batch_size=50)
		)
        self.classifier.initialize()
        layers_info = PrintLayerInfo()
        layers_info(self.classifier)        

    # type 2 CNN with dropout and variable learning rate with 4 conv+4 maxpool
    def define_cnn_type2_varupd_dropout(self):
        sys.setrecursionlimit(10000)
        # with dropout
        self.classifier = NeuralNet(
            layers = [
		('input', layers.InputLayer),
		('conv1', layers.Conv2DLayer),
		('pool1', layers.MaxPool2DLayer),
                ('dropout1', layers.DropoutLayer),
		('conv2', layers.Conv2DLayer),
		('pool2', layers.MaxPool2DLayer),
                ('dropout2', layers.DropoutLayer),
		('conv3', layers.Conv2DLayer),
		('pool3', layers.MaxPool2DLayer),
                ('dropout3', layers.DropoutLayer),
		('conv4', layers.Conv2DLayer),
		('pool4', layers.MaxPool2DLayer),
                ('dropout4', layers.DropoutLayer),
		("hidden5", layers.DenseLayer),
		("output", layers.DenseLayer),
		],
		#layer parameters:
		input_shape = (None, self.img_ch, self.img_height, self.img_width),
		conv1_num_filters = 16, conv1_filter_size = (12, 12), pool1_pool_size = (2,2),
                dropout1_p=0.3,
		conv2_num_filters = 24, conv2_filter_size = (12, 12), pool2_pool_size = (2,2),
                dropout2_p=0.4,
		conv3_num_filters = 32, conv3_filter_size = (12, 12), pool3_pool_size = (2,2),
                dropout3_p=0.5,
		conv4_num_filters = 48, conv4_filter_size = (12, 12), pool4_pool_size = (2,2),
                dropout4_p=0.6,
		hidden5_num_units = 350,
		output_nonlinearity = softmax,
                #NOTE: REMEMBER THAT 1 and 0 are two classes, not one!!
		output_num_units = self.num_class, 

		#optimization parameters:
		update = nesterov_momentum,
                # start the learning rate
                update_learning_rate = theano.shared(float32(0.005)),
                update_momentum = theano.shared(float32(0.9)),
		regression = False,
                # this code implements variable learning rate
                on_epoch_finished=[
                    AdjustVariable('update_learning_rate',start=0.003, stop=0.0001),
                    AdjustVariable('update_momentum', start=0.9, stop=0.999),
                ],
		max_epochs = 300,
		verbose = 2,
                batch_iterator_train=BatchIterator(batch_size=100)
		)
        self.classifier.initialize()
        layers_info = PrintLayerInfo()
        layers_info(self.classifier)        

    ##### INCEPTION v3 related layers #####
    def bn_conv(self,input_layer, **kwargs):
    	l = Conv2DLayer(input_layer, **kwargs)
    	l = batch_norm(l, epsilon=0.001)
    	return l

    def inceptionA(self,input_layer, nfilt):
    	# Corresponds to a modified version of figure 5 in the paper
        l1 = self.bn_conv(input_layer, num_filters=nfilt[0][0], filter_size=1)
    	l2 = self.bn_conv(input_layer, num_filters=nfilt[1][0], filter_size=1)
    	l2 = self.bn_conv(l2, num_filters=nfilt[1][1], filter_size=5, pad=2)
    	l3 = self.bn_conv(input_layer, num_filters=nfilt[2][0], filter_size=1)
   	l3 = self.bn_conv(l3, num_filters=nfilt[2][1], filter_size=3, pad=1)
    	l3 = self.bn_conv(l3, num_filters=nfilt[2][2], filter_size=3, pad=1)
    	l4 = Pool2DLayer(
            input_layer, pool_size=3, stride=1, pad=1, mode='average_exc_pad')
    	l4 = self.bn_conv(l4, num_filters=nfilt[3][0], filter_size=1)
    	return ConcatLayer([l1, l2, l3, l4])

    def inceptionB(self,input_layer, nfilt):
    	# Corresponds to a modified version of figure 10 in the paper
    	l1 = self.bn_conv(input_layer, num_filters=nfilt[0][0], filter_size=3, stride=2)
    	l2 = self.bn_conv(input_layer, num_filters=nfilt[1][0], filter_size=1)
    	l2 = self.bn_conv(l2, num_filters=nfilt[1][1], filter_size=3, pad=1)
    	l2 = self.bn_conv(l2, num_filters=nfilt[1][2], filter_size=3, stride=2)
    	l3 = Pool2DLayer(input_layer, pool_size=3, stride=2)
    	return ConcatLayer([l1, l2, l3])

    def inceptionC(self,input_layer, nfilt):
    	# Corresponds to figure 6 in the paper
    	l1 = self.bn_conv(input_layer, num_filters=nfilt[0][0], filter_size=1)
    	l2 = self.bn_conv(input_layer, num_filters=nfilt[1][0], filter_size=1)
    	l2 = self.bn_conv(l2, num_filters=nfilt[1][1], filter_size=(1, 7), pad=(0, 3))
    	l2 = self.bn_conv(l2, num_filters=nfilt[1][2], filter_size=(7, 1), pad=(3, 0))
    	l3 = self.bn_conv(input_layer, num_filters=nfilt[2][0], filter_size=1)
    	l3 = self.bn_conv(l3, num_filters=nfilt[2][1], filter_size=(7, 1), pad=(3, 0))
    	l3 = self.bn_conv(l3, num_filters=nfilt[2][2], filter_size=(1, 7), pad=(0, 3))
    	l3 = self.bn_conv(l3, num_filters=nfilt[2][3], filter_size=(7, 1), pad=(3, 0))
    	l3 = self.bn_conv(l3, num_filters=nfilt[2][4], filter_size=(1, 7), pad=(0, 3))
    	l4 = Pool2DLayer(
            input_layer, pool_size=3, stride=1, pad=1, mode='average_exc_pad')
    	l4 = self.bn_conv(l4, num_filters=nfilt[3][0], filter_size=1)
    	return ConcatLayer([l1, l2, l3, l4])

    def inceptionD(self,input_layer, nfilt):
    	# Corresponds to a modified version of figure 10 in the paper
    	l1 = self.bn_conv(input_layer, num_filters=nfilt[0][0], filter_size=1)
    	l1 = self.bn_conv(l1, num_filters=nfilt[0][1], filter_size=3, stride=2)
    	l2 = self.bn_conv(input_layer, num_filters=nfilt[1][0], filter_size=1)
    	l2 = self.bn_conv(l2, num_filters=nfilt[1][1], filter_size=(1, 7), pad=(0, 3))
    	l2 = self.bn_conv(l2, num_filters=nfilt[1][2], filter_size=(7, 1), pad=(3, 0))
    	l2 = self.bn_conv(l2, num_filters=nfilt[1][3], filter_size=3, stride=2)
    	l3 = Pool2DLayer(input_layer, pool_size=3, stride=2)
    	return ConcatLayer([l1, l2, l3])

    def inceptionE(self,input_layer, nfilt, pool_mode):
    	# Corresponds to figure 7 in the paper
    	l1 = self.bn_conv(input_layer, num_filters=nfilt[0][0], filter_size=1)
    	l2 = self.bn_conv(input_layer, num_filters=nfilt[1][0], filter_size=1)
    	l2a = self.bn_conv(l2, num_filters=nfilt[1][1], filter_size=(1, 3), pad=(0, 1))
    	l2b = self.bn_conv(l2, num_filters=nfilt[1][2], filter_size=(3, 1), pad=(1, 0))
    	l3 = self.bn_conv(input_layer, num_filters=nfilt[2][0], filter_size=1)
    	l3 = self.bn_conv(l3, num_filters=nfilt[2][1], filter_size=3, pad=1)
    	l3a = self.bn_conv(l3, num_filters=nfilt[2][2], filter_size=(1, 3), pad=(0, 1))
    	l3b = self.bn_conv(l3, num_filters=nfilt[2][3], filter_size=(3, 1), pad=(1, 0))
    	l4 = Pool2DLayer(
            input_layer, pool_size=3, stride=1, pad=1, mode=pool_mode)
    	l4 = self.bn_conv(l4, num_filters=nfilt[3][0], filter_size=1)
        return ConcatLayer([l1, l2a, l2b, l3a, l3b, l4])

    def build_inception_v3_network(self):
        layers = InputLayer(shape=(None, 3, 299, 299))
        layers = self.bn_conv(layers, num_filters=32, filter_size=3, stride=2)
        layers = self.bn_conv(layers, num_filters=32, filter_size=3)
        layers = self.bn_conv(layers, num_filters=64, filter_size=3, pad=1)
        layers = Pool2DLayer(layers, pool_size=3, stride=2, mode='max')
        layers = self.bn_conv(layers, num_filters=80, filter_size=1)
        layers = self.bn_conv(layers, num_filters=192, filter_size=3)
        layers = Pool2DLayer(layers, pool_size=3, stride=2, mode='max')
        layers = self.inceptionA(layers, nfilt=((64,), (48, 64), (64, 96, 96), (32,)))
        layers = self.inceptionA(layers, nfilt=((64,), (48, 64), (64, 96, 96), (64,)))
        layers = self.inceptionA(layers, nfilt=((64,), (48, 64), (64, 96, 96), (64,)))
        layers = self.inceptionB(layers, nfilt=((384,), (64, 96, 96)))
        layers = self.inceptionC(layers, nfilt=((192,), (128, 128, 192), (128, 128, 128, 128, 192), (192,)))
        layers = self.inceptionC(layers, nfilt=((192,), (160, 160, 192), (160, 160, 160, 160, 192), (192,)))
        layers = self.inceptionC(layers, nfilt=((192,), (160, 160, 192), (160, 160, 160, 160, 192), (192,)))
        layers = self.inceptionC(layers, nfilt=((192,), (192, 192, 192), (192, 192, 192, 192, 192), (192,)))
        layers = self.inceptionD(layers, nfilt=((192, 320), (192, 192, 192, 192)))
        layers = self.inceptionE(layers, nfilt=((320,), (384, 384, 384), (448, 384, 384, 384), (192,)),pool_mode='average_exc_pad')
        layers = self.inceptionE(layers, nfilt=((320,), (384, 384, 384), (448, 384, 384, 384), (192,)),pool_mode='max')
        layers = GlobalPoolLayer(layers)
	layers = DropoutLayer(layers, p=0.8)
        layers = DenseLayer(layers, num_units=self.num_class, nonlinearity=None)
        return layers

    #### segmentation networks ####
    def build_recombinator_network_5lyr(self):
        net = {}
        l = InputLayer(shape=(None, self.img_ch, self.img_width,self.img_height))
        net['input'] = l
        l = ConvLayer(l, 32, 3, pad='same')
        net['T4'] = l
        l = ConvLayer(Pool2DLayer(l, 2), 64, 3, pad='same')
        net['T3'] = l
        l = ConvLayer(Pool2DLayer(l, 2), 128, 3, pad='same')
        net['T2'] = l
        l = ConvLayer(Pool2DLayer(l, 2), 256, 3, pad='same')
        net['T1'] = l
        l = ConvLayer(Pool2DLayer(l, 2), 512, 3, pad='same')
        net['T0'] = l
        l = ConvLayer(ConvLayer(net['T0'], 512, 3, pad='same'), 256, 3, pad='same')
        l = Upscale2DLayer(l, 2)
	net['M0'] = l
        l = ConcatLayer((net['T1'], net['M0']))
        l = ConvLayer(ConvLayer(l, 256, 3, pad='same'), 128, 3, pad='same')
        l = Upscale2DLayer(l, 2)
        net['M1'] = l
        l = ConcatLayer((net['T2'], net['M1']))
        l = ConvLayer(ConvLayer(l, 128, 3, pad='same'), 64, 3, pad='same')
        l = Upscale2DLayer(l, 2)
        net['M2'] = l
        l = ConcatLayer((net['T3'], net['M2']))
        l = ConvLayer(ConvLayer(l, 64, 3, pad='same'), 32, 3, pad='same')
        l = Upscale2DLayer(l, 2)
        net['M3'] = l
        l = ConcatLayer((net['T4'], net['M3']))
        l = ConvLayer(ConvLayer(l, 32, 3, pad='same'), 1, 3, pad='same', nonlinearity=sigmoid)
        net['M4'] = l
        return l

    def build_recombinator_network(self):
    	net = {}
    	l = InputLayer(shape=(None, self.img_ch, self.img_width,self.img_height))
    	net['input'] = l
    	l = ConvLayer(l, 16, 3, pad='same')
    	net['T4'] = l
    	l = ConvLayer(Pool2DLayer(l, 2), 32, 3, pad='same')
    	net['T3'] = l
    	l = ConvLayer(Pool2DLayer(l, 2), 48, 3, pad='same')
    	net['T2'] = l
    	l = ConvLayer(Pool2DLayer(l, 2), 48, 3, pad='same')
    	net['T1'] = l
    	l = ConvLayer(ConvLayer(net['T1'], 48, 3, pad='same'), 48, 3, pad='same')
    	l = Upscale2DLayer(l, 2)
    	net['M1'] = l
    	l = ConcatLayer((net['T2'], net['M1']))
    	l = ConvLayer(ConvLayer(l, 48, 3, pad='same'), 32, 3, pad='same')
    	l = Upscale2DLayer(l, 2)
    	net['M2'] = l
    	l = ConcatLayer((net['T3'], net['M2']))
    	l = ConvLayer(ConvLayer(l, 32, 3, pad='same'), 16, 3, pad='same')
    	l = Upscale2DLayer(l, 2)
    	net['M3'] = l
    	l = ConcatLayer((net['T4'], net['M3']))
    	l = ConvLayer(ConvLayer(l, 16, 3, pad='same'), 1, 3, pad='same', nonlinearity=sigmoid)
    	net['M4'] = l
    	return l

    def build_unet(self):
	net = {}
	l = InputLayer(shape=(None, self.img_ch, self.img_width,self.img_height))

	l = ConvLayer(ConvLayer(l, 32, 3, nonlinearity=rectify, pad='same'), 32, 3, nonlinearity=rectify, pad='same')
	net['conv1'] = l

	l = ConvLayer(ConvLayer(Pool2DLayer(l, 2), 64, 3, nonlinearity=rectify, pad='same'), 64, 3, nonlinearity=rectify, pad='same')
	net['conv2'] = l

	l = ConvLayer(ConvLayer(Pool2DLayer(l, 2), 128, 3, nonlinearity=rectify, pad='same'), 128, 3, nonlinearity=rectify, pad='same')
	net['conv3'] = l

	l = ConvLayer(ConvLayer(Pool2DLayer(l, 2), 256, 3, nonlinearity=rectify, pad='same'), 256, 3, nonlinearity=rectify, pad='same')
	net['conv4'] = l

	l = ConvLayer(ConvLayer(Pool2DLayer(l, 2), 512, 3, nonlinearity=rectify, pad='same'), 512, 3, nonlinearity=rectify, pad='same')
	net['conv5'] = l

	l = ConcatLayer((Upscale2DLayer(net['conv5'], 2), net['conv4']),axis=1)
	net['up6'] = l

	l = ConvLayer(ConvLayer(net['up6'], 256, 3, nonlinearity=rectify, pad='same'), 256, 3, nonlinearity=rectify, pad='same')
	net['conv6'] = l

	l = ConcatLayer((Upscale2DLayer(net['conv6'], 2), net['conv3']),axis=1)
	net['up7'] = l

	l = ConvLayer(ConvLayer(net['up7'], 128, 3, nonlinearity=rectify, pad='same'), 128, 3, nonlinearity=rectify, pad='same')
	net['conv7'] = l

	l = ConcatLayer((Upscale2DLayer(net['conv7'], 2), net['conv2']),axis=1)
	net['up8'] = l

	l = ConvLayer(ConvLayer(net['up8'], 64, 3, nonlinearity=rectify, pad='same'), 64, 3, nonlinearity=rectify, pad='same')
	net['conv8'] = l

	l = ConcatLayer((Upscale2DLayer(net['conv8'], 2), net['conv1']),axis=1)
	net['up9'] = l

	l = ConvLayer(ConvLayer(net['up9'], 32, 3, nonlinearity=rectify, pad='same'), 32, 3, nonlinearity=rectify, pad='same')
	net['conv9'] = l

	l = ConvLayer(l, 1, 1, nonlinearity=sigmoid)
	net['conv10'] = l

	return l

    ##### resnet related layers #####
    # expect input of 224x224
    def build_resnet34(self, input_var=None):    
        # create a residual learning building block with two stacked 3x3 convlayers as in paper
        def residual_block(l, increase_dim=False, projection=True):
        	input_num_filters = l.output_shape[1]
        	if increase_dim:
            		first_stride = (2,2)
            		out_num_filters = input_num_filters*2
        	else:
            		first_stride = (1,1)
            		out_num_filters = input_num_filters

        	stack_1 = batch_norm(ConvLayer(l, num_filters=out_num_filters, filter_size=(3,3), stride=first_stride, nonlinearity=rectify, pad='same', W=lasagne.init.HeNormal(gain='relu'), flip_filters=False))
        	stack_2 = batch_norm(ConvLayer(stack_1, num_filters=out_num_filters, filter_size=(3,3), stride=(1,1), nonlinearity=None, pad='same', W=lasagne.init.HeNormal(gain='relu'), flip_filters=False))
        
        	# add shortcut connections
        	if increase_dim:
            		if projection:
                		# projection shortcut, as option B in paper
                		projection = batch_norm(ConvLayer(l, num_filters=out_num_filters, filter_size=(1,1), stride=(2,2), nonlinearity=None, pad='same', b=None, flip_filters=False))
                		block = NonlinearityLayer(ElemwiseSumLayer([stack_2, projection]),nonlinearity=rectify)
            		else:
                		# identity shortcut, as option A in paper
                		identity = ExpressionLayer(l, lambda X: X[:, :, ::2, ::2], lambda s: (s[0], s[1], s[2]//2, s[3]//2))
                		padding = PadLayer(identity, [out_num_filters//4,0,0], batch_ndim=1)
                		block = NonlinearityLayer(ElemwiseSumLayer([stack_2, padding]),nonlinearity=rectify)
        	else:
            		block = NonlinearityLayer(ElemwiseSumLayer([stack_2, l]),nonlinearity=rectify)
        
        	return block

        # see http://arxiv.org/pdf/1512.03385v1.pdf for imagenet details
    	# Building the network
    	l_in = InputLayer(shape=(None, self.img_ch, self.img_width, self.img_height), input_var=input_var)

    	# first layer, output is 64 x 112 x 112
    	l = batch_norm(ConvLayer(l_in, num_filters=64, filter_size=(7,7), stride=(2,2), nonlinearity=rectify, pad='same', W=lasagne.init.HeNormal(gain='relu'), flip_filters=False))
        l = Pool2DLayer(l, pool_size=(3,3), stride=(2,2), mode='max', pad=1)
    
    	# first stack of residual blocks, output is 56x56
    	for _ in range(3):
        	l = residual_block(l)

    	# second stack of residual blocks, output is 28x28
    	l = residual_block(l, increase_dim=True)
    	for _ in range(1,4):
        	l = residual_block(l)

    	# third stack of residual blocks, output is 14x14
    	l = residual_block(l, increase_dim=True)
    	for _ in range(1,6):
        	l = residual_block(l)

        # fourth stack of residual blocks, output is 7x7
    	l = residual_block(l, increase_dim=True)
    	for _ in range(1,3):
        	l = residual_block(l)
    
    	# average pooling
    	l = GlobalPoolLayer(l)

    	# fully connected layer w/ dropout
        l = DropoutLayer(l, p=0.7)
    	network = DenseLayer(l, num_units=self.num_class,nonlinearity=None)

    	return network

    # resnet50 and above
    def build_resnet50_101_152(self, input_var=None,layer_cnt=50):    
        # create a residual learning building block with two stacked 3x3 convlayers as in paper
        def residual_block(l, increase_dim=False, projection=True):
        	input_num_filters = l.output_shape[1]
        	if increase_dim:
            		first_stride = (2,2)
            		out_num_filters = input_num_filters*2
        	else:
            		first_stride = (1,1)
            		out_num_filters = input_num_filters

                # in 50 layer and above, stacks are 1x1,3x3,1x1
        	stack_1 = batch_norm(ConvLayer(l, num_filters=out_num_filters, filter_size=(1,1), stride=first_stride, nonlinearity=rectify, pad='same', W=lasagne.init.HeNormal(gain='relu'), flip_filters=False))
        	stack_2 = batch_norm(ConvLayer(stack_1, num_filters=out_num_filters, filter_size=(3,3), stride=(1,1), nonlinearity=None, pad='same', W=lasagne.init.HeNormal(gain='relu'), flip_filters=False))
        	stack_3 = batch_norm(ConvLayer(stack_2, num_filters=out_num_filters, filter_size=(1,1), stride=(1,1), nonlinearity=None, pad='same', W=lasagne.init.HeNormal(gain='relu'), flip_filters=False))
                
        	# add shortcut connections
        	if increase_dim:
            		if projection:
                		# projection shortcut, as option B in paper
                		projection = batch_norm(ConvLayer(l, num_filters=out_num_filters, filter_size=(1,1), stride=(2,2), nonlinearity=None, pad='same', b=None, flip_filters=False))
                		block = NonlinearityLayer(ElemwiseSumLayer([stack_3, projection]),nonlinearity=rectify)
            		else:
                		# identity shortcut, as option A in paper
                		identity = ExpressionLayer(l, lambda X: X[:, :, ::2, ::2], lambda s: (s[0], s[1], s[2]//2, s[3]//2))
                		padding = PadLayer(identity, [out_num_filters//4,0,0], batch_ndim=1)
                		block = NonlinearityLayer(ElemwiseSumLayer([stack_3, padding]),nonlinearity=rectify)
        	else:
            		block = NonlinearityLayer(ElemwiseSumLayer([stack_3, l]),nonlinearity=rectify)
        
        	return block

        if layer_cnt != 50 and layer_cnt != 101 and layer_cnt != 152:
                print("ERROR: Layer count "+str(layer_cnt)+" is not supported for resnet!")

        # see http://arxiv.org/pdf/1512.03385v1.pdf for imagenet details
    	# Building the network
    	l_in = InputLayer(shape=(None, self.img_ch, self.img_width, self.img_height), input_var=input_var)

    	# first layer, output is 64 x 112 x 112
    	l = batch_norm(ConvLayer(l_in, num_filters=64, filter_size=(7,7), stride=(2,2), nonlinearity=rectify, pad='same', W=lasagne.init.HeNormal(gain='relu'), flip_filters=False))
        l = Pool2DLayer(l, pool_size=(3,3), stride=(2,2), mode='max', pad=1)
    
    	# first stack of residual blocks, output is 56x56
    	for _ in range(3):
        	l = residual_block(l)

        # second stack of residual blocks, output is 28x28
        l = residual_block(l, increase_dim=True)
        if layer_cnt == 50 or layer_cnt == 101:
                for _ in range(1,4):
                    l = residual_block(l)
        else:
                # 152 case
                for _ in range(1,8):
                    l = residual_block(l)
                    
    	# third stack of residual blocks, output is 14x14
    	l = residual_block(l, increase_dim=True)
        if layer_cnt == 50:
                for _ in range(1,6):
                    l = residual_block(l)
        elif layer_cnt == 101:
                for _ in range(1,23):
                    l = residual_block(l)
        else:
                for _ in range(1,36):
                    l = residual_block(l)
                    
        # fourth stack of residual blocks, output is 7x7
    	l = residual_block(l, increase_dim=True)
    	for _ in range(1,3):
        	l = residual_block(l)
    
    	# average pooling
    	l = GlobalPoolLayer(l)

    	# fully connected layer
    	#network = DenseLayer(
        #	l, num_units=self.num_class,
        #    	W=lasagne.init.HeNormal(),
        #    	nonlinearity=None)

        l = DropoutLayer(l, p=0.5)
        network = DenseLayer(l, num_units=self.num_class,nonlinearity=None)

    	return network

    def multilabel_objective(self,predictions, targets):
    	epsilon = np.float32(10e-8)
    	one = np.float32(1.0)
    	output = T.clip(predictions, epsilon, one - epsilon)
    	#return -T.sum(targets * T.log(pred) + (one - targets) * T.log(one - pred), axis=1)
	return T.mean(lasagne.objectives.binary_crossentropy(output.flatten(), targets.flatten()))

    def dice_coef(self,predictions,targets):
	y_true_f = T.flatten(targets,outdim=2)
	y_pred_f = T.flatten(predictions,outdim=2)
	# must be negative!
	return -T.mean((2. * T.dot(y_true_f, T.transpose(y_pred_f)).diagonal() + 1.) / (T.sum(y_true_f,axis=1) + T.sum(y_pred_f,axis=1) + 1.))

################################

# update learning rate class
class AdjustVariable(object):
    def __init__(self, name, start=0.00005, stop=0.000001):
        self.name = name
        self.start, self.stop = start, stop
        self.ls = None

    def __call__(self, nn, train_history):
        if self.ls is None:
            self.ls = np.linspace(self.start, self.stop, nn.max_epochs)

        epoch = train_history[-1]['epoch']
        new_value = float32(self.ls[epoch - 1])
        getattr(nn, self.name).set_value(new_value)

# this adjusts the learning rate by checking the train_loss for the past epoch_limit
# if max-min/max < threshold, reduce learning rate by divisor
class StepVariable(object):
    def __init__(self, name, start=0.001, epoch_limit=10, threshold=0.9, divisor=3.000):
	self.name = name
	self.start, self.epoch_limit, self.threshold, self.divisor = start, epoch_limit, threshold, divisor
	self.learning_rate = start

    def __call__(self, nn, train_history):
	train_loss_max = max(train_history[-1]['train_loss'])
	train_loss_min = min(train_history[-1]['train_loss'])
	train_loss_drop = train_loss_min / train_loss_max

	if train_loss_drop < threshold:
		self.learning_rate = self.learning_rate / self.divisor
		print("Info: Adjusting learning rate to "+str(self.learning_rate))

class AllImageIterator(BatchIterator):

    ''' 
	This is a subclass of BatchIterator by reading multiple numpy files 
	in order to process all images given
    '''

    # add directory variable to original
    def __init__(self, batch_size, shuffle=False, seed=42, cache_dir=None, n_classes=10, augment=False, grayscale=False):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.random = np.random.RandomState(seed)
	self.cache_dir = cache_dir
	self.n_classes = n_classes
	self.augment = augment
	self.grayscale = grayscale

    # generate batches for all images, instead of just a subset 
    def __iter__(self):
	# first look for npy chunks via glob
	cache_chunks = list(glob.iglob(self.cache_dir + "/train_data_inputs*cache.npy"))
	bs = self.batch_size
	for cache_chunk in cache_chunks:
		# load numpy input/target
		# print("Info: Loading current numpy chunk "+cache_chunk)
		cache_target = re.sub(r'train_data_inputs','train_data_targets',cache_chunk)
		self.X = np.load(cache_chunk)
		self.y_target = np.load(cache_target).astype(float32)

		# manipulate matrices so that it fits standard criteria
		if grayscale is False:
			self.X = self.X.reshape(-1,3,np.size(self.X,2),np.size(self.X,3)).astype(float32)
		else:
			self.X = self.X.reshape(-1,1,np.size(self.X,1),np.size(self.X,2)).astype(float32)

		[self.X,self.y_target] = self.shuffle_set(self.X,self.y_target)

		self.y = np.zeros(shape=(np.size(self.y_target),self.n_classes))
		for i in range(0,self.n_classes):
			self.y[self.y_target == i,i] = 1.00
		self.y = self.y.astype(float32)

		# generate usual batches
		for i in range((len(self.X) + bs -1) // bs):
			sl = slice(i * bs, (i + 1) * bs)
			Xb = self._sldict(self.X, sl)
			if self.y is not None:
				yb = self.y[sl]
			else:
				yb = None
			yield self.transform(Xb,yb)

    def _sldict(self,arr, sl):
    	if isinstance(arr, dict):
        	return {k: v[sl] for k, v in arr.items()}
    	else:
        	return arr[sl]

    def shuffle_set(self,X,Y):
        X_shuffle = copy(X)
        Y_shuffle = copy(Y)
        i=0
        rs = cross_validation.ShuffleSplit(size(Y,0), n_iter=1, test_size=0, random_state=0)
        for train_index, test_index in rs:
            for j in train_index:
                X_shuffle[i] = copy(X[j])
                Y_shuffle[i] = copy(Y[j])
                i += 1
        return [X_shuffle, Y_shuffle]


    def transform(self, Xb, yb):
	Xb, yb = super(AllImageIterator, self).transform(Xb, yb)
	if self.augment is False:
		return Xb, yb
	bs = Xb.shape[0]
	indices = np.random.choice(bs, (2 * bs) / 3, replace=False)
	indices_cw = indices[:len(indices)/2]
	indices_ccw = indices[len(indices)/2:]
	Xb_transformed = Xb.copy()
	# Xb will definitely have c01, so feed directly
	for i in indices_ccw:
		#print("Info: Rotating ccw for index="+str(i))
		Xb_transformed[i] = self.im_affine_transform(Xb[i],
							     scale=1.00,
							     rotation=10.0,
							     shear=0.0,
							     translation_x=0.0,
							     translation_y=15.0)
	for i in indices_cw:
		#print("Info: Rotating cw for index="+str(i))
                Xb_transformed[i] = self.im_affine_transform(Xb[i],
                                                             scale=1.00,
                                                             rotation=-10.0,
                                                             shear=0.0,
                                                             translation_x=0.0,
                                                             translation_y=-15.0)
	return Xb_transformed, yb

    def warp(self,img, tf, output_shape, mode='constant', order=0):
    	"""
    	This wrapper function is faster than skimage.transform.warp
   	"""
   	m = tf.params
    	img = img.transpose(2, 0, 1)
    	t_img = np.zeros(img.shape, img.dtype)
    	for i in range(t_img.shape[0]):
        	t_img[i] = _warp_fast(img[i], m, output_shape=output_shape[:2],
                              	      mode=mode, order=order)
    	t_img = t_img.transpose(1, 2, 0)
    	return t_img

    def im_affine_transform(self,img, scale, rotation, shear, translation_y, translation_x, return_tform=False):
    	# Assumed img in c01. Convert to 01c for skimage
    	img = img.transpose(1, 2, 0)
    	# Normalize so that the param acts more like im_rotate, im_translate etc
    	scale = 1 / scale
    	translation_x = - translation_x
    	translation_y = - translation_y

    	# shift to center first so that image is rotated around center
    	center_shift = np.array((img.shape[0], img.shape[1])) / 2. - 0.5
    	tform_center = SimilarityTransform(translation=-center_shift)
    	tform_uncenter = SimilarityTransform(translation=center_shift)

    	rotation = np.deg2rad(rotation)
    	tform = AffineTransform(scale=(scale, scale), rotation=rotation,
        	                shear=shear,
                            	translation=(translation_x, translation_y))
    	tform = tform_center + tform + tform_uncenter

    	warped_img = self.warp(img, tform, output_shape=img.shape)

    	# Convert back from 01c to c01
    	warped_img = warped_img.transpose(2, 0, 1)
    	warped_img = warped_img.astype(img.dtype)
    	if return_tform:
        	return warped_img, tform
    	else:
        	return warped_img
