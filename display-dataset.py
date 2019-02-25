from __future__ import print_function

import os
import sys
import cv2

import numpy as np

from fashion_dataset import load_dataset

dataset = sys.argv[1]

classes_codes = ["tshirt", "trouser","pullover",
                "dress","coat","sandal","shirt",
                "sneaker","bag","boot"]

(x_val, y_val) = load_dataset(sys.argv[1])

GRAY = [180,180,180]
BLACK = [0,0,0]
idx = 0
print(len(x_val))
while True:

    img = x_val[idx]
    class_num = y_val[idx]
    
    square_img = cv2.copyMakeBorder(img,20,60,80,80,cv2.BORDER_CONSTANT,value=GRAY)

    real_idx = idx if idx >= 0 else (len(x_val)+idx)
    cv2.putText(square_img, str(real_idx) + ": " + classes_codes[class_num] + "(" + str(class_num) + ")", 
                    (5,90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, BLACK, 2)

    cv2.imshow("img", square_img)

    key = cv2.waitKey(0)
    print(key)
    if key == 27:   # ESC
        break
    elif key == 83: #right
        idx += 1
        if idx == len(x_val):
            idx = 0
    elif key == 81:  #left
        idx -= 1
        if idx == -len(x_val):
            idx = 0
    elif key == 115:  # "s" (save)
        name = classes_codes[class_num] + "-" + str(class_num) + "_idx-" + str(real_idx) + ".png"
        print("Saving image as " + name)
        cv2.imwrite(name, img)


