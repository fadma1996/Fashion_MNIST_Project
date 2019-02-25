from __future__ import print_function
import keras
from keras.models import load_model

import numpy as np
import os
import cv2
import sys

import conv
from skimage import io

#"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def test_image(img_name,model):

	classes = ["T-shirt/top", "Trouser","Pullover",
		    "Dress","Coat","Sandal","Shirt",
		    "Sneaker","Bag","Ankle boot"]
	img = io.imread(img_name)
	print("img.shape", img.shape)

	# Pre-process
	converted = conv.convert(img)

	# write this for debug only
	cv2.imwrite("converted.png", converted)

	# Normalize data
	x = converted.reshape((28,28,1))

	x = x.astype('float32')
	x/=255
	x=np.expand_dims(x, axis=0)

	# load model and predict

	

	predicted_x = model.predict(x)

	pred_class = np.argmax(predicted_x)
	print('Class:', pred_class, classes[pred_class])
        
	#NEW: sort and pick the best five
	pairs = list(enumerate(predicted_x[0]))
	pairs.sort(key=lambda x:-x[1])
	top5 = pairs[:5]
	array_top5=[]
        for i, res in enumerate(top5): 
             msg ="{}: {}({}) {:.2f}%".format(i, classes[res[0]], res[0], res[1]*100)
             array_top5.append(msg)
             
	return array_top5
# Formatted results
#for i, res in enumerate(top5): 
 #   msg ="{}: {}({}) {:.2f}%".format(i, classes[res[0]], res[0], res[1]*100)
  #  print(msg)


