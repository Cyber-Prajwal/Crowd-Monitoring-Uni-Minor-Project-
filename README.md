# Crowd-Monitoring-Uni-Minor-Project-
With this Python program we attempted to study the effect of our university's flap gates on student movement, and how it affects the same. With the program you can monitor number of people, how long they stay in the frame, their path trace until they leave the frame and display a "CROWDED!!" alert if its getting too crowded.
 You can also monitor cars, dogs, bikes, busses etc, by just changing a few parameters in the code.

## Here are the results obtained :
#### Flap Gates

https://github.com/user-attachments/assets/9abf98a5-53d5-451c-8f1e-8b0d4f2fbbb3

<br>

#### Dog tracking 

https://github.com/user-attachments/assets/1fea094c-0c3c-494d-a355-7ed38c4495c8

https://github.com/user-attachments/assets/638175f6-5fca-4bda-8446-06d1b79a8960

<br>

#### Tracking on crosswalk

https://github.com/user-attachments/assets/63d08642-2dd3-4f08-9bad-817988b7f4ec

#### Tracking cars

https://github.com/user-attachments/assets/dcb4b4f0-deed-4ef6-ba36-43ca4d66a75f

https://github.com/user-attachments/assets/d6151d34-cc6a-4b57-adcb-15c31ae5a705

<br><br>

This Python program studies the effect of university flap gates on student movement. Here's a step-by-step breakdown of the code implementation:

1. **Setting up the environment:**
   - The code uses libraries like OpenCV, NumPy, imutils, and scipy.
   - MobileNetSSD is used for object detection, loaded with:
     ```python
     detector = cv2.dnn.readNetFromCaffe(prototxt=protopath, caffeModel=modelpath)
     ```

2. **Implementing the CentroidTracker:**
   - A custom `CentroidTracker` class is defined in `centroidtracker.py`.
   - It keeps track of objects using OrderedDict to store object IDs, centroids, and bounding boxes.
   - Methods like `register()`, `deregister()`, and `update()` manage object tracking.

3. **Video capture and processing loop:**
   - Video is captured using OpenCV's VideoCapture:
     ```python
     cap = cv2.VideoCapture('flap_final.mp4')
     ```
   - The main loop processes each frame:
     ```python
     while True:
         ret, frame = cap.read()
         frame = imutils.resize(frame, width=600)
     ```

4. **Object detection:**
   - Each frame is processed through the MobileNetSSD network:
     ```python
     blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
     detector.setInput(blob)
     person_detections = detector.forward()
     ```
   - Detections are filtered based on confidence and class (person):
     ```python
     if confidence > 0.5 and CLASSES[index] == "person":
         # Process detection
     ```

5. **Non-maximum suppression:**
   - The `non_max_suppression_fast()` function is applied to remove overlapping detections.

6. **Updating object tracking:**
   - The CentroidTracker is updated with new detections:
     ```python
     objects = tracker.update(rects)
     ```

7. **Drawing bounding boxes and traces:**
   - For each tracked object, bounding boxes are drawn:
     ```python
     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
     ```
   - Centroids are stored and used to draw path traces:
     ```python
     centroid_dict[objectId].append((cX, cY))
     # ... (code for drawing lines)
     ```

8. **Calculating and displaying dwell time:**
   - Dwell time for each object is calculated and displayed:
     ```python
     dwell_time[objectId] += sec
     text = "ID:{}|time:{}".format(objectId, int(dwell_time[objectId]))
     cv2.putText(frame, text, (x1, y1-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)
     ```

9. **Implementing crowding detection:**
   - The number of tracked objects is monitored:
     ```python
     if object_id_list[-1] > 7:
         cv2.putText(frame, crowded_text, (5, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 0), 2)
     ```

10. **Displaying results:**
    - Processed frames are displayed using OpenCV:
      ```python
      cv2.imshow("Application", frame)
      ```

11. **Handling user input:**
    - The program exits when 'q' is pressed:
      ```python
      if key == ord('q'):
          break
      ```

This implementation allows for real-time monitoring of crowd movement, tracking individual persons, and alerting when crowding occurs. The code can be adapted for other objects by modifying the `CLASSES` list and adjusting detection parameters.

## Running the Program

1. Ensure all required libraries are installed:
   ```
   pip install opencv-python numpy imutils scipy
   ```
2. Prepare your video source (file or camera)
3. Run the script:
   ```
   python draw_tracking_line.py
   ```
4. Press 'q' to quit the application

Note: Adjust parameters like `maxDisappeared` and `maxDistance` in the CentroidTracker initialization to fine-tune the tracking performance for your specific use case.

## Customization

To adapt the program for monitoring other objects:
1. Modify the `CLASSES` list in the code
2. Adjust the confidence threshold in the main loop
3. Update the object index in the detection loop (e.g., change "person" to "car")

## Requirements

- Python 3.x
- OpenCV
- NumPy
- imutils
- scipy
