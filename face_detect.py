#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 10:55:24 2020

@author: Aniket
"""

import numpy as np
import cv2

#print(cv2.__file__)
ana_dir = '/data/'
cv_path = ana_dir+'anaconda3/share/OpenCV/haarcascades/'
face_cascade = cv2.CascadeClassifier(cv_path+'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv_path+'haarcascade_eye.xml')

img_file = 'test.jpeg'
img = cv2.imread(img_file)
if img is None : 
    print('Image not read')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray, 1.3, 5)
for (x,y,w,h) in faces:
    img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()