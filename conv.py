import numpy as np
import os
import cv2
import sys

from skimage import exposure

# trim white border
def trim(base_img):

    white_pixels = base_img > 245

    white_cols = np.all(white_pixels, axis=0)
    white_rows = np.all(white_pixels, axis=1)

    good_rows = np.where(white_rows==False)[0]
    good_cols = np.where(white_cols==False)[0]

    y_min = good_rows[0]
    y_max = good_rows[-1]

    x_min = good_cols[0]
    x_max = good_cols[-1]

    return base_img[y_min:y_max+1, x_min:x_max+1]


def detect_background(img):

    mask = img > 252
    #print(mask)

    #img[mask]=0
    #cv2.imshow("masked", img)
    #cv2.waitKey(0)

    #exit(0)

    return mask
   

def to_grayscale(img):

    gray_img = img.copy()

    if len(img.shape) == 2 or img.shape[2] == 1:
        
        gray_img = img.reshape(img.shape[0],img.shape[1],1)

    elif img.shape[2] == 3: # to grayscale

        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    elif img.shape[2] == 4:   # convert transparent background to white

        alpha_channel=img[:,:,3]
        background_mask=alpha_channel < 10   # select transparent pixels

        # convert to grayscale
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Make white a little darker (so it's different from background)
        gray_img[gray_img > 240] -=10

        # Make all the transparent background white
        gray_img[background_mask] = 255

    return gray_img

def convert(img):

    proc_img = to_grayscale(img)

    # trim extra border
    proc_img = trim(proc_img)

    background_mask = detect_background(proc_img)
    
    # invert
    proc_img = cv2.bitwise_not(proc_img).reshape((proc_img.shape[0],proc_img.shape[1],1))
    
    # set background to black
    proc_img[background_mask] = 0   #BLACK
    
    # add border to make it squared
    size_diff = abs(proc_img.shape[0] - proc_img.shape[1])
    one_side = int(size_diff/2)
    other_side = size_diff - one_side
    BLACK = [0,0,0]
    #print("size_diff", size_diff)
    if proc_img.shape[0] > proc_img.shape[1]:  # tall image
        square_img = cv2.copyMakeBorder(proc_img,0,0,one_side,other_side,cv2.BORDER_CONSTANT,value=BLACK)
    else:   #long image
        square_img = cv2.copyMakeBorder(proc_img,one_side,other_side,0,0,cv2.BORDER_CONSTANT,value=BLACK)

    # downscale
    small_img = cv2.resize(square_img,(28,28), interpolation = cv2.INTER_AREA)
    
    return small_img


if __name__ == '__main__':

    from skimage import io
    img = io.imread(sys.argv[1])

    #img = cv2.imread(sys.argv[1], cv2.IMREAD_UNCHANGED) # we need to keep the alpha channel (R,G,B + alpha)
    print("img.shape", img.shape)

    img = convert(img)
    cv2.imwrite("okk.jpg", img)


