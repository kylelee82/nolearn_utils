#!/usr/bin/python
from CNN_JPG_Classifier import CNN_JPG_Classifier
from numpy import *
from sklearn import cross_validation
import sys
import re
sys.setrecursionlimit(50000)
# return a shuffled training or test set
def shuffle_set(X,Y):
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

###############################
# set options
driver_cnn = CNN_JPG_Classifier()
driver_cnn.set_img_zoom(0,450,50,640)
driver_cnn.set_verbosity(True)
driver_cnn.set_grayscale(False)
driver_cnn.augment_training(False)
driver_cnn.set_resize(True,299,299)
driver_cnn.set_num_classes(10)
driver_cnn.set_max_train(5000)
driver_cnn.set_zoom_mode(False)
driver_cnn.set_pretrain_model(True)
driver_cnn.set_pretrain_name("inception")
###############################

###############################
# data loading
# 1. read training directories
# classes are 1,0,2,3, etc.
# 0 = safe driving
# 1 = texting - right
# 2 = phone - right
# 3 = texting - left
# 4 = phone - left
# 5 = radio
# 6 = drinking
# 7 = reaching behind
# 8 = makeup
# 9 = talking
#driver_cnn.read_train_dirs_all(pics_dir=["kaggle/statefarm/train/c0","kaggle/statefarm/train/c1","kaggle/statefarm/train/c2","kaggle/statefarm/train/c3","kaggle/statefarm/train/c4","kaggle/statefarm/train/c5","kaggle/statefarm/train/c6",
#                          "kaggle/statefarm/train/c7","kaggle/statefarm/train/c8","kaggle/statefarm/train/c9"],output_dir="./data/kaggle/statefarm/inception_cache/")
# 2. already ran once, just load the training data
driver_cnn.load_train("./kaggle/statefarm/inception_cache/train_data_inputs.3.cache.npy","./kaggle/statefarm/inception_cache/train_data_targets.3.cache.npy")
###############################

print("Info: Total number of training examples:"+str(size(driver_cnn.X_train,0)))

###########################
# define CNN architecture (see file for details)
#driver_cnn.define_cnn_type1()
driver_cnn.define_cnn_inception_v3()
# use pretrained model
# output layer is wrong, but it is OK to retrain that layer
#driver_cnn.load_model("inception_v3.params")
driver_cnn.load_model("driver_cnn.driver_10class.299x299.inception_v3.color.params")
#driver_cnn.load_model("driver_cnn.driver_10class.224x224.vgg19.color.params")
#driver_cnn.load_model("driver_cnn.driver_10class.224x224.vgg19.color.params")

from lasagne.layers import *
layer_cnt=0
#for param in driver_cnn.classifier.layers_:
#    l = driver_cnn.classifier.layers_[layer_cnt]
#    layer_cnt += 1
    #print get_all_params(l)
#    if re.search(r'(conv|hidden)',param):
#        l.params[l.W].remove('trainable')
#        l.params[l.b].remove('trainable')
#        print("Info: Removed trainability status from layer "+str(param)+"!")
###########################

###########################
# fit training data (Xtrain needs to be S x 1 x X x Y)
# training data as float
X_train = driver_cnn.X_train[:,:,:]
# change 3 -> 1 for grayscale, and 2,3 to 1,2
X_train = X_train.reshape(-1,3,size(X_train,2),size(X_train,3)).astype(float32)
# change label data to int32 for classifier
#Y_train = driver_cnn.Y_train.astype(int32)
# change label data to float32 for regression
Y_train = driver_cnn.Y_train.astype(int32)
###########################

# shuffle both X and Y so that the fit will contain random labels in order
[X_train,Y_train] = shuffle_set(X_train,Y_train)
#driver_cnn.show_image_dataset(X_train[10,0,:,:])
#print(Y_train[10])

# for Y, for regression, need to put it as a m x n matrix where n = # of classes
Y_regress = zeros(shape=(size(driver_cnn.Y_train),10))
for i in range(0,10):
    Y_regress[Y_train == i,i] = 1.00

Y_regress = Y_regress.astype(float32)

#1.classification
#driver_cnn.fit(X_train,Y_train)
#2.regression
driver_cnn.fit(X_train,Y_regress)

# save the model
# for type1_b architecture
driver_cnn.save_model("driver_cnn.driver_10class.299x299.inception_v3.color.params")

