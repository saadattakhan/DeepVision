import cv2
import h5py
import multiprocessing
from multiprocessing import Queue, Pool
from scipy.spatial import KDTree
import numpy as np
import scipy.misc
import dlib
import pandas as pd


def worker(input_q, output_q):
	while True:
		frame,dets = input_q.get()
		output_q.put(frame)

def load_dlib_models(predictor_path="shape_predictor_68_face_landmarks.dat",model_path="dlib_face_recognition_resnet_model_v1.dat"):
	sp = dlib.shape_predictor(predictor_path)
	facerec = dlib.face_recognition_model_v1(model_path)
	detector = dlib.get_frontal_face_detector()	

	return (detector,sp,facerec)

def compute_KDTREE(landmarks):
	return KDTree(landmarks)

def query_KDTREE(tree,test):
	return tree.query(test)[1]

def load_landmarks_labels(file_path="centroids.h5"):
	'''hf = h5py.File(file_path,'r')
	landmarks=hf["landmarks"]
	labels=hf["labels"]'''
	df = pd.read_hdf('faceRec.h5')
	landmarks = np.array(df.iloc[:,:128])
	labels = df.iloc[:,128]

	return (landmarks,labels)

		
if __name__ == "__main__":
	cap=cv2.VideoCapture("video/27-Sept-2017-09-10.mp4")
	input_q = Queue(maxsize=5)
	output_q = Queue(maxsize=5)

	detector,sp,facerec=load_dlib_models()
	landmarks,labels=load_landmarks_labels()
	tree=compute_KDTREE(landmarks)
	


	n_threads=multiprocessing.cpu_count()
	faceTrackers = {}
	faceScores = {}
	rectangleColor = (0,165,255)
	frame_count=0


	pool = Pool(n_threads, worker, (input_q, output_q))



	while True:
		ret,frame=cap.read()
		frame_count+=1

		fidsToDelete = []
		for fid in faceTrackers.keys():
			trackingQuality = faceTrackers[ fid ].update( frame )

			#If the tracking quality is good enough, we must delete
			#this tracker
			if trackingQuality < 7:
				fidsToDelete.append( fid )

		for fid in fidsToDelete:
			print("Removing fid " + str(fid) + " from list of trackers")
			faceTrackers.pop( fid , None )




		dets=None
		if frame_count % 12 ==0:
			dets = detector(frame, 0)
			for k, d in enumerate(dets):
				shape = sp(frame, d)
				test_descriptor = facerec.compute_face_descriptor(frame, shape)
				pred_id=query_KDTREE(tree,test_descriptor)
				pred_label=labels[pred_id]
				pred_descriptor=landmarks[pred_id]
				dist=np.linalg.norm(np.array(test_descriptor)-pred_descriptor)
				print "prediction: "+pred_label+" distance: "+str(dist)
				if dist < 0.45:
					print "Predicted: "+pred_label
					x = int(d.left())
					y = int(d.top())
					w = int(d.width())
					h = int(d.height())


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

						print("Creating new tracker " + pred_label)

						tracker = dlib.correlation_tracker()
						tracker.start_track(frame,
											dlib.rectangle( x-10,
															y-20,
															x+w+10,
															y+h+20))

						faceTrackers[ pred_label ] = tracker
						faceScores[pred_label] = dist
					else:
						if dist<faceScores[matchedFid]:
							print("Removing fid " + str(fid) + " from list of trackers")
							faceTrackers.pop( matchedFid , None )
							faceScores.pop( matchedFid , None)

							tracker = dlib.correlation_tracker()
							tracker.start_track(frame,
												dlib.rectangle( x-10,
																y-20,
																x+w+10,
																y+h+20))

							faceTrackers[ pred_label ] = tracker
							faceScores[pred_label] = dist

		input_q.put((frame,dets))
		if output_q.empty():
			pass
		else:
			output_frame=output_q.get()
			for fid in faceTrackers.keys():
				tracked_position =  faceTrackers[fid].get_position()

				t_x = int(tracked_position.left())
				t_y = int(tracked_position.top())
				t_w = int(tracked_position.width())
				t_h = int(tracked_position.height())

				cv2.rectangle(output_frame, (t_x, t_y),
										(t_x + t_w , t_y + t_h),
										rectangleColor ,2)


				cv2.putText(output_frame, fid ,
								(int(t_x + t_w/2), int(t_y)), 
								cv2.FONT_HERSHEY_SIMPLEX,
								0.5, (255, 255, 255), 2)

			cv2.imshow('Video', scipy.misc.imresize(output_frame, (480,640)))
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break
	pool.terminate()
	cap.release()
	cv2.destroyAllWindows()