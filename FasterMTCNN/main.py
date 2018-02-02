import tensorflow as tf
import numpy as np
import cv2
import os
from os.path import join as pjoin
import sys
import copy
from utils import detect_face
from utils import nn4 as network
import random
import imutils


import sklearn

import scipy.misc
from datetime import datetime
import time

from imutils.video import FileVideoStream
from imutils.video import FPS


def to_rgb(img):
  w, h = img.shape
  ret = np.empty((w, h, 3), dtype=np.uint8)
  ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
  return ret

def detect(frame,pnet,rnet,onet):

	scaleWidth=frame.shape[1]/450
	frame = imutils.resize(frame, width=450)
	result_frame=frame.copy()
	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	frame = np.dstack([frame, frame, frame])


	images_dir="images/"
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
		crop=result_frame[face_position[1]:face_position[3],face_position[0]:face_position[2]]
		cv2.imwrite(images_dir+str(datetime.now())+".png",crop)                       
		cv2.rectangle(result_frame, (face_position[0],face_position[1]),(face_position[2], face_position[3]),(0, 255, 0), 1)
	return result_frame
	

if __name__ == '__main__':
	
	cam="videos/test.avi"
	print("[INFO] starting video file thread...")
	fvs = FileVideoStream(cam).start()
	time.sleep(1.0)

	fps = FPS().start()
	frame_interval=12 # frame intervals  

	numFrames=0
	with tf.Graph().as_default():
		sess = tf.Session()
		with sess.as_default():										
			pnet, rnet, onet = detect_face.create_mtcnn(sess, './model_check_point/')	
		while True:
			frame = fvs.read()
			numFrames=numFrames+1			
			if(numFrames%frame_interval == 0):
				output_frame=(detect(frame,pnet,rnet,onet)).copy()
			else:
				output_frame=frame.copy()
			cv2.imshow('Video', scipy.misc.imresize(output_frame, (480,640)))
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break	
			fps.update()		
		fps.stop()
		print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
		print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
		 
		# do a bit of cleanup
		cv2.destroyAllWindows()
		fvs.stop()
	sess.close()