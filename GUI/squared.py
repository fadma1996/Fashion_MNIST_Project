
def make_squared(image):

    import numpy as np
    import cv2

    proc_img = np.array(image)
    print("proc_img.shape", proc_img.shape)‏
    size_diff = abs(proc_img.shape[0] - proc_img.shape[1])
    one_side = int(size_diff/2)
    other_side = size_diff - one_side
    BG_COLOR = proc_img[0,0].tolist()   # take color from the image background
    if proc_img.shape[0] > proc_img.shape[1]:  # tall image
        square_img = cv2.copyMakeBorder(proc_img,0,0,one_side,other_side,cv2.BORDER_CONSTANT,value=BG_COLOR)
    else:   #long image
        square_img = cv2.copyMakeBorder(proc_img,one_side,other_side,0,0,cv2.BORDER_CONSTANT,value=BG_COLOR)

    squared_image = Image.fromarray(square_img)
    squared_image = squared_image.resize((300,300), Image.ANTIALIAS)
    photo = ImageTk.PhotoImage(squared_image)

    return photo
