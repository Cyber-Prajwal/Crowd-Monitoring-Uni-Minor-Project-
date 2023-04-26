import random
import cv2
import datetime
import imutils
import numpy as np
from centroidtracker import CentroidTracker
from collections import defaultdict

protopath = "MobileNetSSD_deploy.prototxt"
modelpath = "MobileNetSSD_deploy.caffemodel"
detector = cv2.dnn.readNetFromCaffe(prototxt=protopath, caffeModel=modelpath)

tracker = CentroidTracker(maxDisappeared=80, maxDistance=90)


def non_max_suppression_fast(boxes, overlapThresh):
    try:
        if len(boxes) == 0:
            return []

        if boxes.dtype.kind == "i":
            boxes = boxes.astype("float")

        pick = []

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        indexs = np.argsort(y2)

        while len(indexs) > 0:
            last = len(indexs) - 1
            i = indexs[last]
            pick.append(i)

            xx1 = np.maximum(x1[i], x1[indexs[:last]])
            yy1 = np.maximum(y1[i], y1[indexs[:last]])
            xx2 = np.minimum(x2[i], x2[indexs[:last]])
            yy2 = np.minimum(y2[i], y2[indexs[:last]])

            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)

            overlap = (w * h) / area[indexs[:last]]

            indexs = np.delete(indexs, np.concatenate(([last],
                                                   np.where(overlap > overlapThresh)[0])))

        return boxes[pick].astype("int")
    except Exception as e:
        print("Exception occurred in non_max_suppression : {}".format(e))

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

def main():
    cap = cv2.VideoCapture('flap_final.mp4') #to read frames from a video file 
    #we can also do live with webacam if required by entering 1 for usb camera and 0 for inbuilt camera instead of video file

    fps_start_time = datetime.datetime.now()
    fps = 0
    total_frames = 0
    centroid_dict = defaultdict(list)
    object_id_list = []
    dtime = dict()
    dwell_time = dict()

    while True:
        ret, frame = cap.read()
        frame = imutils.resize(frame, width=600) #imutils used to resize the video file
        total_frames = total_frames + 1

        (H, W) = frame.shape[:2] #Height and Width

        blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5) #used for inferencing 

        detector.setInput(blob)  #our detector 
        person_detections = detector.forward()
        rects = []
        for i in np.arange(0, person_detections.shape[2]):
            
            confidence = person_detections[0, 0, i, 2]
            if confidence > 0.5:
                index = int(person_detections[0, 0, i, 1])

                if CLASSES[index] != "person":
                    continue

                person_box = person_detections[0, 0, i, 3:7] * np.array([W, H, W, H]) #creating bounding box
                (startX, startY, endX, endY) = person_box.astype("int")
                rects.append(person_box)

        boundingboxes = np.array(rects)
        boundingboxes = boundingboxes.astype(int)
        rects = non_max_suppression_fast(boundingboxes, 0.3)

        objects = tracker.update(rects)
        for (objectId, bbox) in objects.items():
            x1, y1, x2, y2 = bbox
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)

            cX = int((x1 + x2) / 2.0)
            cY = int((y1 + y2) / 2.0)
            cv2.circle(frame, (cX, cY), 4, (0, 255, 0), -1)

            centroid_dict[objectId].append((cX, cY))
            if objectId not in object_id_list:
                object_id_list.append(objectId)
                start_pt = (cX, cY)
                end_pt = (cX, cY) #coordinates
                cv2.line(frame, start_pt, end_pt, (0, 255, 0), 2)
                dtime[objectId] = datetime.datetime.now()
                dwell_time[objectId] = 0
                
            else:
                curr_time = datetime.datetime.now()
                old_time = dtime[objectId]
                time_diff = curr_time - old_time
                dtime[objectId] = datetime.datetime.now()
                sec = time_diff.total_seconds()
                dwell_time[objectId] += sec
                l = len(centroid_dict[objectId])
                for pt in range(len(centroid_dict[objectId])):
                    if not pt + 1 == l:
                        start_pt = (centroid_dict[objectId][pt][0], centroid_dict[objectId][pt][1])
                        end_pt = (centroid_dict[objectId][pt + 1][0], centroid_dict[objectId][pt + 1][1])
                        cv2.line(frame, start_pt, end_pt, (0, 255, 0), 2)
            #COLORS = (np.random.randint(0,255), np.random.randint(0,255), np.random.randint(0,255))
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            #cv2.rectangle(frame, (x1, y1), (x2, y2), COLORS, 2) #rectangle surrounding the students
            text = "ID:{}|time:{}".format(objectId, int(dwell_time[objectId]))
            cv2.putText(frame, text, (x1, y1-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1) #font and font color

        fps_end_time = datetime.datetime.now()
        time_diff = fps_end_time - fps_start_time
        if time_diff.seconds == 0:
            fps = 0.0
        else:
            fps = (total_frames / time_diff.seconds)

        fps_text = "FPS: {:.2f}".format(fps)
        crowded_text = "CROWDED !!"

        cv2.putText(frame, fps_text, (5, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)
        if object_id_list[-1] > 7:
            cv2.putText(frame, crowded_text, (5, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 0), 2)

        cv2.imshow("Application", frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    cv2.destroyAllWindows()


main()
#Drawing tracking line:
#This script draws a line denoting where the person has entered in the frame and where he has moved in the frame. 