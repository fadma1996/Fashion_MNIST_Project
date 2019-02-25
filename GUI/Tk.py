from Tkinter import *
from PIL import ImageTk, Image
import Tkinter, Tkconstants, tkFileDialog
import testimg
import keras
from keras.models import load_model

from os.path import join
from glob import glob
import numpy as np
import cv2
import numpy as np
import cv2
 
def make_squared(image):

    proc_img = np.array(image) 
    size_diff = abs(proc_img.shape[0] - proc_img.shape[1])
    one_side = int(size_diff/2)
    other_side = size_diff - one_side
    BG_COLOR = proc_img[0,0].tolist()    
    if proc_img.shape[0] > proc_img.shape[1]: 
        square_img = cv2.copyMakeBorder(proc_img,0,0,one_side,other_side,cv2.BORDER_CONSTANT,value=BG_COLOR)
    else:   
        square_img = cv2.copyMakeBorder(proc_img,one_side,other_side,0,0,cv2.BORDER_CONSTANT,value=BG_COLOR)

    squared_image = Image.fromarray(square_img)
    squared_image = squared_image.resize((300,300), Image.ANTIALIAS)
    photo = ImageTk.PhotoImage(squared_image)

    return photo  
def hide_me(event):
    event.widget.pack_forget() 
def read_folder_images():
	files = []
	for ext in ('*.gif', '*.png', '*.jpg','*.jpeg'):
	   files.extend(glob(join("load/", ext)))
	return files 
def move(delta):
    Canevas.delete(ALL) 
    global current, image_list 
        
    if not (0 < current + delta < len(image_list)):
        print('End', 'No more image.')
        return
 
    image_list= read_folder_images()
    x=len(image_list)-1 
    if current==2 and delta==-1 :
        back_btn.pack_forget()

    if delta==1 and current>=0:
        back_btn.pack() 
    if current==x-1 :
        next_btn.pack_forget()

    if delta==-1 and current<=x:
        next_btn.pack() 
    current += delta
    image = Image.open(image_list[current])
    photo = ImageTk.PhotoImage(image) 

    predict_img(image_list[current])
def returnPicture(pic, resize=""):
    img = Image.open(pic)
    if resize:
        img = img.resize(resize, Image.ANTIALIAS)
    photo = ImageTk.PhotoImage(img)
    return photo

def predict_img(filename):
    image = Image.open(filename)
    photo = ImageTk.PhotoImage(image)
    photo = make_squared(image)
    gifdict[filename] = photo  # 
    Canevas.create_image(0,0,anchor=NW,image=photo)
    Canevas.config(height=photo.height(),width=photo.width())
    array_top5=testimg.test_image(filename,model)
    Label_text1.set(array_top5[0])
    Label_text2.set(array_top5[1])
    Label_text3.set(array_top5[2])
    Label_text4.set(array_top5[3]) 

def Ouvrir(): 
    ftypes = [('all files', '.png .jpeg .jpg'),('jpg files', '.jpg'), ('jpeg files', '.jpeg')]
    filename =tkFileDialog.askopenfilename(initialdir = "/home/maryem/Downloads/project4/GUI/load",title = "Select file",filetypes =ftypes)
    im = Image.open(filename)
    photo = PhotoImage(im) 
    predict_img(filename)
 
image_list= read_folder_images()
text_list= read_folder_images()
current = 4
 
root = Tk()
root.title('Fashion-mnist')
root.resizable(width=False,height=False)
root.geometry("1100x700")
#Center
photo= returnPicture("imagesGUI/e.png", resize=(1100,700))
center=Label(root, image = photo)
center.pack(side="top", fill="both", expand="yes", padx="0", pady="0")

#Gray
Gray = Frame(center,borderwidth=2,relief=GROOVE)
Gray.pack(side=TOP,padx=10,pady=10)

#---------------------
model=load_model("model.h5")
#---------------------
p= returnPicture("imagesGUI/a.jpeg", resize=(500,150))
Label(Gray,image=p).pack(side="top", padx="10", pady="10")

btn= Button(Gray, text ='Select an image',font=("Arial", 14, "bold"),command= Ouvrir).pack(side="top")
 
Fram1 = Frame(Gray,relief=GROOVE)
Fram1.pack(side=TOP,padx=10,pady=10)
 
Fram_back = Frame(Fram1,borderwidth=0,relief=GROOVE)
Fram_back.pack(side=LEFT,padx=10,pady=10)
 
Fram2 = Frame(Fram1,borderwidth=4,highlightbackground="gray", highlightthickness=2, bd= 0)
Fram2.pack(side=LEFT,padx=10,pady=10)
 
Fram_next = Frame(Fram1,borderwidth=0,relief=GROOVE)
Fram_next.pack(side=LEFT,padx=10,pady=10)

back_img= returnPicture("imagesGUI/back.png", resize=(50,50))
back_btn= Button(Fram_back,border=0,image =back_img,command=lambda: move(-1))
back_btn.pack(side="left") 
gifdict={} 
default_img= returnPicture("imagesGUI/select_img.png", resize=(200,300))
Canevas = Canvas(Fram2 ,width =200, height =300)

gifdict["imagesGUI/next.png"] = default_img    

    
Canevas.create_image(0,0,anchor=NW,image=default_img)
Canevas.config(height=default_img.height(),width=default_img.width())

Canevas.pack(side="left") 
next_img= returnPicture("imagesGUI/next.png", resize=(50,50))
next_btn= Button(Fram_next,border=0,image =next_img,command=lambda: move(+1))
next_btn.pack(side="left") 
Label_text1 =StringVar()
Label_text1.set("1.-------")
Label_extra1 = Label(Gray,textvariable=Label_text1,font=("Arial", 14, "bold")).pack()

Label_text2 =StringVar()
Label_text2.set("2.-------")
Label_extra2 = Label(Gray,textvariable=Label_text2,font=("Arial", 14, "bold")).pack()

Label_text3=StringVar()
Label_text3.set("3.-------")
Label_extra3 = Label(Gray,textvariable=Label_text3,font=("Arial", 14, "bold")).pack()

Label_text4=StringVar()
Label_text4.set("4.-------")
Label_extra4 = Label(Gray,textvariable=Label_text4,font=("Arial", 14, "bold")).pack()


Label_espace = Label(Gray,text="",font=("Arial", 14, "bold")).pack()  
root.mainloop()
