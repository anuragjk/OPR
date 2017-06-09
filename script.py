import cv2
from matplotlib import pyplot as plt
import numpy as np

img1 = cv2.imread('testimage.jpg',0)

blur = cv2.GaussianBlur(img1,(5,5),0)
thr = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,75,10)
invrt = cv2.bitwise_not(thr)
im2, contours, hierarchy = cv2.findContours(invrt,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#lines = cv2.HoughLines(thr,1,np.pi/180,80)


for i in contours:
    area = cv2.contourArea(i)
    if(area>50000):
        x,y,w,h = cv2.boundingRect(i)
        img = cv2.rectangle(thr,(x,y),(x+w,y+h),(0,255,0),2)
        crop_img = img[y:y+h, x:x+w]



plt.subplot(121),plt.imshow(img1,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(crop_img,cmap = 'gray')
plt.title('Output Image'), plt.xticks([]), plt.yticks([])
