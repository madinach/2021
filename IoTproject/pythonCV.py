# import the necessary packages
import numpy as np
import argparse
from matplotlib import pyplot as plt
plt.rcParams["figure.figsize"] = (20,10)
import cv2
import numpy as np

import pandas as pd
import json
import sys

upper,left,lower,right,middle= 0,0,0,0,0

img= []
number_bolts,bearing_color, axle_color =0,0,0
image= cv2.imread(cv2.samples.findFile("noflash2.jpeg"))
image= image[400:1200, 200:900]
#image = cv2.resize(image, (1200, 700))
# Read image.
img = image
  
# Convert to grayscale.
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  
# Blur using 3 * 3 kernel.
gray_blurred = cv2.blur(gray, (13, 13))
  
# Apply Hough transform on the blurred image.
detected_circles = cv2.HoughCircles(gray_blurred, 
                   cv2.HOUGH_GRADIENT, 1, 20, param1 = 50,
               param2 = 30, minRadius = 34, maxRadius = 45)
if detected_circles is None: 
    number_bolts=0
    #print(number_bolts)
# Draw circles that are detected.
if detected_circles is not None:
    # Convert the circle parameters a, b and r to integers.
    detected_circles = np.uint64(np.around(detected_circles))
    

#number_bolts= len(detected_circles[0])

#print(len(detected_circles[0]))
detected_circles = detected_circles.astype('int64')

for index1, circle1 in enumerate(detected_circles[0]):
    for circle2 in detected_circles[0][index1+1:]:
        point1, Radius1 = circle1[:2], circle1[2]
        point2, Radius2 = circle2[:2], circle2[2]
        #collision or containment:
        if np.linalg.norm(point1-point2) < Radius1 + Radius2:
            print(circle1,circle2)

for circle in detected_circles[0]:
    x,y= circle[0], circle[1]
    #vertically located bolts
    if x<=500 and x>=300: 
        if y<=530 and y>=300: 
            middle= [x,y]
        else: 
            #determine upper/lower
            #upper
            #if y
            if y<=400: 
                upper=1
                #print(x,y, 'upper')
                #print('upper', upper)
            else: 
                lower=1
                #print('lower', lower)
    else: 
        if x<=400: 
            left=1
            #print('left', left)
        else: 
            right=1
            #print('right', right)
    for pt in detected_circles[0, :]:
        a, b, r = pt[0], pt[1], pt[2]
  
        # Draw the circumference of the circle.
        cv2.circle(img, (a, b), r, (0, 255, 0), 2)
  
        # Draw a small circle (of radius 1) to show the center.
        cv2.circle(img, (a, b), 1, (0, 0, 255), 3)
        cv2.namedWindow("Detected Circle", cv2.WINDOW_NORMAL)
        cv2.imshow("Detected Circle", image)
        
        if cv2.waitKey(30) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            cv2.waitKey(0)
        plt.imshow(image)
        
#color detection 

x= 400
y= 308
# Convert to grayscale.
#gray = cv2.circle(gray, (x,y), radius=0, color=(255,255,255), thickness=5)
color = int(gray[x, y])


# 0-none, 1- white, 2-gray, 3-black
if color<50: 
    bearing_color= 3
elif color>140: 
    bearing_color= 1
else: 
    if color<100:
        bearing_color= 0
    else: 
        bearing_color= 2
        
#0-none, 3- black, 2-gray middle[0]+4,middle[1]      
if middle: 
    color2 = int(gray[400, 400])
    if color2<0: 
        axle_color= 3
    elif color2>140: 
        axle_color= 2
    else: 
        if color2<100:
            axle_color= 0
    
 '''       
print(color)
print('left', left,'right', right)
print('upper', upper, 'lower', lower)
print('middle', middle)
print('bearing', bearing_color)    
print('axle', axle_color) 
'''

for a in sys.argv[1:]:
    myDictionary={"value": [upper,left,right,lower,bearing_color,axle_color,'square']}
    print(json.dumps(myDictionary))