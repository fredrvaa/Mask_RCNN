import cv2
import numpy as np
import os

# define range of red color in HSV 160-180 and 0-20
lower_red = np.array([0,50,50])
upper_red = np.array([11,255,255])
lower_red2 = np.array([173,50,50])
upper_red2 = np.array([179,255,255])

ROOT_DIR = os.path.abspath("../")
IMAGES_PATH = os.path.join(ROOT_DIR, "images/images")
PRED_PATH = os.path.join(ROOT_DIR, "images/pred_images")

for file in os.listdir(IMAGES_PATH):
    image = cv2.imread("{}/{}".format(IMAGES_PATH, file))

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Threshold the HSV image to get only red colors
    mask1 = cv2.inRange(hsv, lower_red, upper_red)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask=mask1+mask2
    ret,maskbin = cv2.threshold(mask, 127,255,cv2.THRESH_BINARY) 
    cv2.imshow('maskbin', mask)
    cv2.imshow('image', image)
    cv2.waitKey(0)
