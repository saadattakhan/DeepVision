import tensorflow as tf
import numpy as np
import cv2
import os
from os.path import join as pjoin
import sys
import copy
import detect_face
import nn4 as network
import random


import sklearn

from sklearn.externals import joblib
import scipy.misc

import multiprocessing
from multiprocessing import Queue, Pool

import time



def to_rgb(img):
  w, h = img.shape
  ret = np.empty((w, h, 3), dtype=np.uint8)
  ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
  return ret


def detect(frame,pnet,rnet,onet):
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	minsize = 20 # minimum size of face
	threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
	factor = 0.709
	if gray.ndim == 2:
		img = to_rgb(gray)
	bounding_boxes, _ = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
	nrof_faces = bounding_boxes.shape[0]
	for face_position in bounding_boxes:            
		face_position=face_position.astype(int)                       
		cv2.rectangle(frame, (face_position[0],face_position[1]),(face_position[2], face_position[3]),(0, 255, 0), 2)
	return frame

def worker(input_q, output_q):
	with tf.Graph().as_default():
		sess = tf.Session()
		with sess.as_default():
			pnet, rnet, onet = detect_face.create_mtcnn(sess, './model_check_point/')
		while True:
			frame = input_q.get()
			output_q.put(detect(frame,pnet,rnet,onet))
	sess.close()

if __name__ == '__main__':
	input_q = Queue(maxsize=5)
	output_q = Queue(maxsize=5)
	cam="rtsp://admin:123@dmin123@10.16.0.101:554/cam/realmonitor?channel=5&subtype=0"
	video_capture=cv2.VideoCapture(cam)
	c=0
	frame_interval=2 # frame intervals  
	pool = Pool(2, worker, (input_q, output_q))	 # scale factor

	start=time.time()
	max_frames = 50;
	numFrames=0
	
	while True:
		ret, frame = video_capture.read()
		if ret:
			

			timeF = frame_interval
			if(c%timeF == 0):
				numFrames=numFrames+1
				input_q.put(frame)
				if output_q.empty():
					pass
				else:
					cv2.imshow('Video', scipy.misc.imresize(output_q.get(), (480,640)))
					if numFrames == max_frames:
						end = time.time()
						seconds = end - start
						newfps  = numFrames / seconds;
						print "Estimated frames per second : {0}".format(newfps);
					
			c+=1
			#cv2.imshow('Video', scipy.misc.imresize(frame, (480,640)))
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break
    


	pool.terminate()
	video_capture.release()
	cv2.destroyAllWindows()