# -*- coding: utf-8 -*-
import cv2, os, glob, time, sys
import numpy as np
import tensorflow as tf
import base64, glob#, cStringIO

from utils import load_weights, detections_boxes
from yolo_v3_gpu import yolo_v3


class_names = 'plate_c.names'
weights_file = 'yolov3_20000.weights'
size = 416


def imgs(x):
      cv2.imshow('Rotat', np.array(x))
      cv2.waitKey(0)
      cv2.destroyAllWindows()

def load_coco_names(file_name):
    names = {}
    with open(file_name) as f:
        for id, name in enumerate(f):
            names[id] = name
    return names
_COLOR = [8,136,19]#[255, 127, 0] Blue


def convert_to_original_size(box, size, original_size):
    ratio = original_size / size
    box = box.reshape(2, 2) * ratio
    return list(box.reshape(-1))

classes = load_coco_names(class_names)
#inputs = tf.placeholder(tf.float32, [ size, size, 3])
inputs = tf.placeholder(tf.float32, [None, size, size, 3])


with tf.variable_scope('detector2'):
        detections, cl_de = yolo_v3(inputs, len(classes), data_format='NHWC')
        load_ops = load_weights(tf.global_variables(scope='detector2'), weights_file)

#boxes = detections_boxes(detections)
boxes, attrs = detections_boxes(detections)

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.30)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
sess.run(load_ops)


def py_nms(boxes, scores, max_boxes=50, iou_thresh=0.5):
    """
    Pure Python NMS baseline.
    Arguments: boxes => shape of [-1, 4], the value of '-1' means that dont know the
                        exact number of boxes
               scores => shape of [-1,]
               max_boxes => representing the maximum of boxes to be selected by non_max_suppression
               iou_thresh => representing iou_threshold for deciding to keep boxes
    """
    assert boxes.shape[1] == 4 and len(scores.shape) == 1

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= iou_thresh)[0]
        order = order[inds + 1]

    return keep[:max_boxes]

def cpu_nms(boxes, scores, num_classes, max_boxes=50, score_thresh=0.4, iou_thresh=0.5):
    """
    /*----------------------------------- NMS on cpu ---------------------------------------*/
    Arguments:
        boxes ==> shape [1, 10647, 4]
        scores ==> shape [1, 10647, num_classes]
    """
    #print boxes
    boxes = boxes.reshape(-1, 4)
    scores = scores.reshape(-1, num_classes)
    # Picked bounding boxes
    picked_boxes, picked_score, picked_label = [], [], []

    for i in range(num_classes):
        indices = np.where(scores[:,i] >= score_thresh)
        filter_boxes = boxes[indices]
        filter_scores = scores[:,i][indices]
        if len(filter_boxes) == 0: continue
        # do non_max_suppression on the cpu
        indices = py_nms(filter_boxes, filter_scores,
                         max_boxes=max_boxes, iou_thresh=iou_thresh)
        picked_boxes.append(filter_boxes[indices])
        picked_score.append(filter_scores[indices])
        picked_label.append(np.ones(len(indices), dtype='int32')*i)
    if len(picked_boxes) == 0: return [], [], []

    boxes = np.concatenate(picked_boxes, axis=0)
    score = np.concatenate(picked_score, axis=0)
    label = np.concatenate(picked_label, axis=0)

    return boxes, score, label




def draw_boxes(image, boxes, scores, labels, classes, detection_size, img):
    """
    :param boxes, shape of  [num, 4]
    :param scores, shape of [num, ]
    :param labels, shape of [num, ]
    :param image,
    :param classes, the return list from the function `read_coco_names`
    """
    #new = np.ones(shape=image.shape, dtype=np.float32) #np.zeros(shape=image.shape, dtype=np.float32)
    if boxes is None: 
        #print "NONE"
        return image, None
    answ = []
    cord = []
    for i in range(len(labels)): # for each bounding box, do:
        bbox, score, label = boxes[i], scores[i], classes[labels[i]]
        bbox_text = "%s %.2f" %(label, score)
        print (bbox)
        coord = [abs(int(x)) for x in bbox]
        o0 = coord[0]
        o1 = coord[1]
        o2 = coord[2]
        o3 = coord[3]
        
        o0 = o0*(img.shape[1] / 416.0)
        o1 = o1*(img.shape[0] / 416.0)
        o2 = o2*(img.shape[1] / 416.0)
        o3 = o3*(img.shape[0] / 416.0)


#        print (x1, y1, x2, y2)
#        image[o1-3:o3, o0:o0+3, :] = _COLOR   # y
#        image[o1-3:o3, o2:o2+3, :] = _COLOR #255  # y

#        image[o3-3:o3, o0:o2+3, :] = _COLOR  # x
#        image[o1-3:o1, o0:o2+3, :] = _COLOR  # x

        img = cv2.rectangle(img, (int(o0), int(o1)), (int(o2), int(o3)), (0,255,0), 3)
        img = cv2.putText(img, bbox_text, (int(o0), int(o1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        
    return img, coord, answ #
    #return None, None, None

#import ocr_s as ocr
def gg(img):
        if img.shape[0] != 100:

                img_resized0 = cv2.resize(img, (size, size)) #img.resize(size=(size, size))
                img_resized = np.reshape(img_resized0, [1, 416, 416, 3])
                I, B, C = sess.run([inputs, boxes, cl_de], feed_dict={inputs: img_resized})
                boxeS, scores, labels = cpu_nms(B, C, len(classes), max_boxes=3000, score_thresh=0.08, iou_thresh=0.09)
                #boxeS, scores, labels = cpu_nms(B, C, len(classes), max_boxes=3000, score_thresh=0.4, iou_thresh=0.5)
                print (boxeS, scores, labels)
#                if scores == []:
#                   coord = []
#                   answ = []                   
#                else:
                image, coord, answ = draw_boxes(img_resized0, boxeS, scores, labels, classes, 416, img)

#                #image, answ_sw, coord_sw = ocr.ocr_img(img)
#                retval, image = cv2.imencode('.jpg', np.array(image))
#                #print (coord_sw)
                return image, coord, answ, scores

class record():
     def __init__(self):
         self.file = open('ERROR.txt', 'w')
         #print "LOAD"
     def save(self, i):
         #print i
         self.file.write(i+"\n") 

"""
def saveImage(list, name, factor, cl):
        with open("RELabels/"+name+".txt", 'w') as f:
            #f.write('%d\n' %len(self.bboxList))
            for ix, bbox in enumerate(list):
                f.write("{} {} {} {} {}\n".format(int(cl[ix]),
                                                  int(int(bbox[0])*factor),
                                                  int(int(bbox[1])*factor),
                                                  int(int(bbox[2])*factor),
                                                  int(int(bbox[3])*factor)))#bbox[4]))
"""
def saveImage(list, name, factor, cl):
        with open("RELabels/"+name+".txt", 'w') as f:
            #f.write('%d\n' %len(self.bboxList))
            for ix, bbox in enumerate(list):
               if int(cl[ix]) == 3 or int(cl[ix]) == 1 or int(cl[ix]) == 2: #if int(cl[ix]) == 3 or int(cl[ix]) == 1:
                        f.write("{} {} {} {} {}\n".format(int(cl[ix]),
                                                          int(int(bbox[0])*factor),
                                                          int(int(bbox[1])*factor),
                                                          int(int(bbox[2])*factor),
                                                          int(int(bbox[3])*factor)))#bbox[4]))
if __name__ == "__main__":
   #print "START"
       iop = cv2.imread("/media/sadko/1b32d2c7-3fcf-4c94-ad20-4fb130a7a7d4/DataTrain/srplate/fe334_T210OA35.png")
       #imgs(iop)
       P = gg(iop)
       imgs(P[0])
