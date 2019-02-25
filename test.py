from __future__ import print_function
import keras
from keras.models import load_model

import numpy as np
import os
import cv2
import sys
import glob
import shutil
import traceback

import conv
from skimage import io

from fashion_dataset import class_from_filename

#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"] = ""

classes = ["T-shirt/top", "Trouser","Pullover",
            "Dress","Coat","Sandal","Shirt",
            "Sneaker","Bag","Ankle boot"]

def glob_filetypes(root_dir, *patterns):
    return [path
            for pattern in patterns
            for path in glob.glob(os.path.join(root_dir, pattern))]

def ensure_dir(file_path):
    if not os.path.exists(file_path):
        os.mkdir(file_path)

ensure_dir("errors/")

model = None


def test_image(img_name,model_name):

    global model

    img = io.imread(img_name)
    print("img.shape", img.shape)


    converted = conv.convert(img)

    x = converted.reshape((28,28,1))

    x = x.astype('float32')
    x/=255
    x=np.expand_dims(x, axis=0)

    if not model:  

        model = load_model(model_name)

    predicted_x = model.predict(x)

    print('!!prediction-x!!',predicted_x)

    pred_class = np.argmax(predicted_x)
    print('Class:', pred_class, classes[pred_class])


    pairs = list(enumerate(predicted_x[0]))
    pairs.sort(key=lambda x:-x[1])
    top5 = pairs[:5]
    print(top5)

    expected_num, expected_class = class_from_filename(img_name)
    print("Expected:", expected_class)

    pred_class_num=top5[0][0]
    if expected_num == pred_class_num:
        print("Match ok:", expected_class, img_name)
    else:
        perc = top5[0][1]*100
        msg = "Match failed: expected {}({}) was {}({})({:.2f}%) for file {}".format(
                                classes[expected_num],expected_num,
                                classes[pred_class_num],pred_class_num,
                                perc,img_name)
        print(msg)


        basename = os.path.basename(img_name)
        no_ext_name, ext = os.path.splitext(basename)

        err_name = "errors/" + no_ext_name + "_as_" + classes[pred_class_num] + "-" + str(int(perc)) + ext
        shutil.copy(img_name, err_name)

        conv_err_name = "errors/" + no_ext_name + "_as_" + classes[pred_class_num] + "-" + str(int(perc)) + "_converted.png"
        #shutil.copy("converted.png", conv_err_name)
        cv2.imwrite(conv_err_name, converted)


    for i, res in enumerate(top5): 
        msg ="{}: {}({}) {:.2f}%".format(i, classes[res[0]], res[0], res[1]*100)
        print(msg)

    return expected_num, pred_class_num

if __name__ == '__main__':

    model_name=sys.argv[1]
    img_name=sys.argv[2]

    if os.path.isdir(img_name):
        files_grabbed = glob_filetypes(img_name, '*.jpeg', '*.jpg', '*.png', '*.JPEG', '*.JPG', '*.PNG')
        print("Found {} files in folder {}".format(len(files_grabbed), img_name))
        Y = []
        y_pred = []
        for f in files_grabbed:
            try:
                expected_num, pred_class_num = test_image(f,model_name)
                Y.append(expected_num)
                y_pred.append(pred_class_num)
            except:
                print("Unexpected error:", sys.exc_info()[0])
                traceback.print_exc()

        print("len", len(Y))

        import confusion_matrix as cm
        import matplotlib.pyplot as plt

        # Compute confusion matrix
        from sklearn.metrics import accuracy_score, confusion_matrix

        accuracy = accuracy_score(Y, y_pred)

        cnf_matrix = confusion_matrix(Y, y_pred)
        np.set_printoptions(precision=2)

        # Plot non-normalized confusion matrix
        plt.figure()
        cm.plot_confusion_matrix(cnf_matrix, classes, accuracy,
                                    normalize=True, title='Confusion matrix')

        plt.show()

    else:
        test_image(img_name,model_name)
    


