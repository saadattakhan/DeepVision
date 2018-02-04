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

import dlib

def to_rgb(img):
  w, h = img.shape
  ret = np.empty((w, h, 3), dtype=np.uint8)
  ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
  return ret

def saveImage(image):
	images_dir="images/"
	images_area=96*96
	if ((image.shape[0]*image.shape[1])>images_area):
		cv2.imwrite(images_dir+str(datetime.now())+".png",image)


def get_point(x,y,w,h,nw,nh):
	

	new_x=(float(x)/float(w))*float(nw)
	new_y=(float(y)/float(h))*float(nh)
	return (int(new_x),int(new_y))

def detect(frame,pnet,rnet,onet,faceTrackers):
	big_image=frame.copy()
	big_height=big_image.shape[0]
	big_width=big_image.shape[1]


	small_image=imutils.resize(frame, width=450)
	small_image = cv2.cvtColor(small_image, cv2.COLOR_BGR2GRAY)
	small_image = np.dstack([small_image, small_image, small_image])

	small_height=small_image.shape[0]
	small_width=small_image.shape[1]


	gray = cv2.cvtColor(small_image, cv2.COLOR_BGR2GRAY)
	minsize = 20 # minimum size of face
	threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
	factor = 0.709
	if gray.ndim == 2:
		img = to_rgb(gray)
	bounding_boxes, _ = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
	nrof_faces = bounding_boxes.shape[0]
	for face_position in bounding_boxes:            
		face_position=face_position.astype(int)


		left_top=get_point(face_position[0],face_position[1],small_width,small_height,big_width,big_height)
		right_bottom=get_point(face_position[2],face_position[3],small_width,small_height,big_width,big_height)


		x=left_top[0]
		y=left_top[1]
		w=right_bottom[0]-left_top[0]
		h=right_bottom[1]-left_top[1] 

		x_bar = x + 0.5 * w
		y_bar = y + 0.5 * h

		matchedFid = None


		for fid in faceTrackers.keys():
			tracked_position =  faceTrackers[fid].get_position()

			t_x = int(tracked_position.left())
			t_y = int(tracked_position.top())
			t_w = int(tracked_position.width())
			t_h = int(tracked_position.height())


			t_x_bar = t_x + 0.5 * t_w
			t_y_bar = t_y + 0.5 * t_h

			if ( ( t_x <= x_bar   <= (t_x + t_w)) and 
				 ( t_y <= y_bar   <= (t_y + t_h)) and 
				 ( x   <= t_x_bar <= (x   + w  )) and 
				 ( y   <= t_y_bar <= (y   + h  ))):
				matchedFid = fid
		if matchedFid is None:
			print("Creating new tracker ")

			tracker = dlib.correlation_tracker()
			tracker.start_track(big_image,
								dlib.rectangle( x-20,
												y-20,
												x+w+20,
												y+h+20))

			faceTrackers[ len(faceTrackers) ] = tracker		
	return (faceTrackers,big_image)
	

if __name__ == '__main__':
	
	cam="videos/test.avi"
	

	print("[INFO] starting video file thread...")
	fvs = FileVideoStream(cam).start()
	time.sleep(1.0)

	fps = FPS().start()
	frame_interval=12 # frame intervals  

	faceTrackers = {}
	rectangleColor = (0,165,255)

	numFrames=0
	with tf.Graph().as_default():
		sess = tf.Session()
		with sess.as_default():										
			pnet, rnet, onet = detect_face.create_mtcnn(sess, './model_check_point/')	
		while True:
			frame = fvs.read()
			numFrames=numFrames+1

			fidsToDelete = []
			for fid in faceTrackers.keys():
				trackingQuality = faceTrackers[ fid ].update( frame )
				if trackingQuality < 7:
					fidsToDelete.append( fid )

			for fid in fidsToDelete:
				print("Removing fid " + str(fid) + " from list of trackers")
				faceTrackers.pop( fid , None )




			if(numFrames%frame_interval == 0):
				faceTrackers,output_frame=detect(frame,pnet,rnet,onet,faceTrackers)
			else:
				output_frame=frame
			for fid in faceTrackers.keys():
				tracked_position =  faceTrackers[fid].get_position()

				t_x = int(tracked_position.left())
				t_y = int(tracked_position.top())
				t_w = int(tracked_position.width())
				t_h = int(tracked_position.height())

				crop=output_frame[t_y:t_y+t_h,t_x:t_x+t_w]
				saveImage(crop) 


				cv2.rectangle(output_frame, (t_x, t_y),
										(t_x + t_w , t_y + t_h),
										rectangleColor ,2)


			cv2.imshow('Video', scipy.misc.imresize(output_frame, (720,1280)))
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