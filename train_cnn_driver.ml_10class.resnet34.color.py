#!/usr/bin/python
from CNN_JPG_Classifier import CNN_JPG_Classifier
from numpy import *
from sklearn import cross_validation
import sys
import re
import lasagne
sys.setrecursionlimit(50000)
import os

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
driver_cnn.set_resize(True,224,224)
driver_cnn.set_num_classes(10)
driver_cnn.set_max_train(12000)
driver_cnn.set_zoom_mode(False)
driver_cnn.set_pretrain_model(False)
driver_cnn.set_pretrain_name("resnet")
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
#driver_cnn.read_train_dirs("kaggle/statefarm/train/c1","kaggle/statefarm/train/c0",
#                         ["kaggle/statefarm/train/c2","kaggle/statefarm/train/c3","kaggle/statefarm/train/c4","kaggle/statefarm/train/c5","kaggle/statefarm/train/c6",
#                          "kaggle/statefarm/train/c7","kaggle/statefarm/train/c8","kaggle/statefarm/train/c9"])
# 2. already ran once, just load the training data
driver_cnn.load_train("kaggle/statefarm/train/c1/train_data_inputs.cache.npy","kaggle/statefarm/train/c1/train_data_targets.cache.npy")
###############################

print("Info: Total number of training examples:"+str(size(driver_cnn.X_train,0)))

###########################
# define CNN architecture (see file for details)
driver_cnn.define_cnn_resnet34()
# use pretrained model

if os.path.isfile('driver_cnn.driver_10class.224x224.resnet34.color.npz'):
	with load('driver_cnn.driver_10class.224x224.resnet34.color.npz') as f:
		param_values = [f['arr_%d' % i] for i in range(len(f.files))]
	lasagne.layers.set_all_param_values(driver_cnn.classifier.layers, param_values)
	print("Info: Loaded parameters from driver_cnn.driver_10class.224x224.resnet34.color.npz")

#from lasagne.layers import *
#layer_cnt=0
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
[X_random,Y_random] = shuffle_set(X_train,Y_train)
#driver_cnn.show_image_dataset(X_random[10,0,:,:])
#print(Y_random[10])

# for Y, for regression, need to put it as a m x n matrix where n = # of classes
Y_regress = zeros(shape=(size(driver_cnn.Y_train),10))
for i in range(0,10):
    Y_regress[Y_random == i,i] = 1.00

Y_regress = Y_regress.astype(float32)

#1.classification
#driver_cnn.fit(X_random,Y_random)
#2.regression
driver_cnn.fit(X_random,Y_regress)

# save the model
# for type1_b architecture
#driver_cnn.save_model("driver_cnn.driver_10class.224x224.resnet34.color.params")

print("Info: Saving model to driver_cnn.driver_10class.224x224.resnet34.color.npz")
savez('driver_cnn.driver_10class.224x224.resnet34.color.npz', *lasagne.layers.get_all_param_values(driver_cnn.classifier.layers))

print("Info: Testing training accuracy...")
Y_val = driver_cnn.predict(X_random)
acc = 100.000 * sum(argmax(Y_val,axis=1) == argmax(Y_regress,axis=1)) / size(argmax(Y_val,axis=1))
print("Info: Training acc = "+str(acc)+"%")
