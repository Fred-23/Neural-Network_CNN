# packages
import os, sys
import cv2
from PIL import Image
import numpy as np
from resizeimage import resizeimage

#image directory path--CHANGE FOR YOUR USAGE
IMG_DIR = 'C:/Users/freda/OneDrive/Bureau/ALL/1-ETS/Images_Samples/Handwritten'


size = (28,28)

# init a canvas
canvas = np.zeros((244, 244, 1), np.uint8)

# make canvas white
canvas.fill(0)

# global coordinates and drawing state
x = 0
y = 0
drawing = False

# mouse callback function
def draw(event, current_x, current_y, flags, params):
    # hook up global variables
    global x, y, drawing
    
    # handle mouse down event
    if event == cv2.EVENT_LBUTTONDOWN:
        # update coordinates
        x = current_x
        y = current_y
        
        # enable drawing flag
        drawing = True
    
    # handle mouse move event
    elif event == cv2.EVENT_MOUSEMOVE:
        # draw only if mouse is down
        if drawing:
            # draw the line
            cv2.line(canvas, (current_x, current_y), (x, y), 255, thickness=3)
            
            # update coordinates
            x, y = current_x, current_y
    
    # handle mouse up event
    elif event == cv2.EVENT_LBUTTONUP:
        # disable drawing flag
        drawing = False
    

# display the canvas in a window
cv2.imshow('Draw', canvas)

# bind mouse events
cv2.setMouseCallback('Draw', draw)

# infinite drawing loop
while True:
    # update canvas
    cv2.imshow('Draw', canvas)
    
    # break out of a program on 'Esc' button hit
    if cv2.waitKey(1) & 0xFF == 27: 
        # PIL image can be saved as .png .jpg .gif or .bmp file (among others)
        #filename = "my_drawing.png"
        #canvas = cv2.resize(crop, dsize=size, interpolation=cv2.INTER_CUBIC)
        
    
        
        isWritten = cv2.imwrite(IMG_DIR+'/handwritten_numbers.png', canvas)
        img = Image.open(IMG_DIR+'/handwritten_numbers.png')
        img = resizeimage.resize_contain(img, size)
        img.save(IMG_DIR+'/handwritten_numbers_resize.png', img.format)
        break




# clean up windows
cv2.destroyAllWindows()