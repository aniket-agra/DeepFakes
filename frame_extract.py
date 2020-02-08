#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 15:49:38 2020

@author: Aniket
"""

import numpy as np
import cv2
import glob
import json
#import string

ana_dir_ofc = '/data'
ana_dir_mac = '/Users/Aniket'
basedir_ofc = '/data/Kaggle/DeepFake/'
basedir_mac = '/Users/Aniket/KaggleData/DeepFake/'
#get all video filenames
files = glob.glob(basedir_ofc+'sample_data/train_sample_videos/*.mp4')


cv_path = ana_dir_ofc+'/anaconda3/share/OpenCV/haarcascades/'
face_cascade = cv2.CascadeClassifier(cv_path+'haarcascade_frontalface_default.xml')
#read json file 
json_file = basedir_ofc+'sample_data/train_sample_videos/metadata.json'
with open(json_file, 'r') as f:
    json_dict = json.load(f)

nvids = len(files)

#loop over videos - capture image info as vectors
for i in np.arange(0, 1) :     
    #start capturing images 
    print('Capturing frames from : {}'.format(files[i]))
    
    cap = cv2.VideoCapture(files[i]) #basedir+'sample_data/train_sample_videos/bmehkyanbj.mp4'
    c = 0
    while(cap.isOpened()):
        ret, frame = cap.read()
        #frame = np.asarray(frame, dtype=np.uint8)
        #print(frame)
        if frame is not None : 
            c = c+1
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x,y,w,h) in faces:
                gray = cv2.rectangle(gray,(x,y),(x+w,y+h),(255,0,0),2)
            cv2.imshow('frame', gray)
            if cv2.waitKey(0) & 0xFF == ord('q'):
                break
        else : 
            break

    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    #video name to look up label in dict
    print('no of frames = {}'.format(c))
    vid_fname = files[0].rsplit("/")[-1]