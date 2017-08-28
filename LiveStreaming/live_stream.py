import cv2
import multiprocessing
from multiprocessing import Queue, Pool
import scipy.misc




def worker(input_q, output_q):
    # Load a (frozen) Tensorflow model into memory.
    while True:
        frame = input_q.get()
        output_q.put(frame)

if __name__ == '__main__':
	cam="rtsp://admin:123@dmin123@10.16.0.101:554/cam/realmonitor?channel=1&subtype=0"
	cap=cv2.VideoCapture(cam)
	input_q = Queue(maxsize=5)
	output_q = Queue(maxsize=5)
	pool = Pool(2, worker, (input_q, output_q))

	while True:
		ret,frame = cap.read()
		if ret:
			input_q.put(frame)
			cv2.imshow("LiveStream", scipy.misc.imresize(output_q.get(), (480,640)))
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break;
	pool.terminate()
	cap.release()
	cv2.destroyAllWindows()