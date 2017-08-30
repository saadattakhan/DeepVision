import os
import cv2
import numpy as np
import tensorflow as tf

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
import scipy.misc
import time




PATH_TO_LABELS = 'object_detection/data/mscoco_label_map.pbtxt'

NUM_CLASSES = 90

# Loading label map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)


def detect_objects(image_np, sess, detection_graph):
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

    # Each box represents a part of the image where a particular object was detected.
    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    scores = detection_graph.get_tensor_by_name('detection_scores:0')
    classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    # Actual detection.
    (boxes, scores, classes, num_detections) = sess.run(
        [boxes, scores, classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})

    # Visualization of the results of a detection.
    print boxes[0][0]
    print image_np.shape
    ymin, xmin, ymax, xmax = boxes[0][5]
    im_width, im_height ,channel = image_np.shape
    (left, right, top, bottom) = (int(xmin * im_width), int(xmax * im_width),int(ymin * im_height), int(ymax * im_height))
    cropped = image_np[top:bottom, left:right]
    cv2.imwrite("result.jpg",cropped)
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=8)
    return image_np




if __name__ == '__main__':
    cam="rtsp://admin:123@dmin123@10.16.0.101:554/cam/realmonitor?channel=5&subtype=0"
    cap=cv2.VideoCapture(0)

    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile("object_detection/ssd_mobilenet_v1_coco_11_06_2017/frozen_inference_graph.pb", 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

        sess = tf.Session(graph=detection_graph)




    start=time.time()
    max_frames = 50;
    numFrames=0

    while True:
        ret,frame = cap.read()

        if ret:
            numFrames=numFrames+1
            cv2.imshow('Video', scipy.misc.imresize(detect_objects(frame, sess, detection_graph), (480,640)))
            if numFrames == max_frames:
                end = time.time()
                seconds = end - start
                newfps  = numFrames / seconds;
                print "Estimated frames per second : {0}".format(newfps);
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break;
    
    sess.close()
    cap.release()
    cv2.destroyAllWindows()
    
    



