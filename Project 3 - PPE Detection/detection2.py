import csv
from datetime import datetime
from ultralytics import YOLO
import cv2
import cvzone
import math

# Open the CSV file for writing
with open('detection_data.csv', mode='w') as detection_file:
    detection_writer = csv.writer(detection_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    detection_writer.writerow(['Class Name', 'Confidence', 'Frame Timestamp'])

cap = cv2.VideoCapture("../Videos/ppe-3.mp4")  # For Video
model = YOLO("ppe.pt")
classNames = ['Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person', 'Safety Cone',
              'Safety Vest', 'machinery', 'vehicle']
myColor = (0, 0, 255)

while True:
    success, img = cap.read()
    results = model(img, stream=True)
    frame_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1

            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100

            # Class Name
            cls = int(box.cls[0])
            currentClass = classNames[cls]
            print(currentClass)
            if conf > 0.5:
                if currentClass == 'NO-Hardhat' or currentClass == 'NO-Safety Vest' or currentClass == "NO-Mask":
                    myColor = (0, 0, 255)
                    # Write detection data to CSV file
                    with open('detection_data.csv', mode='a') as detection_file:
                        detection_writer = csv.writer(detection_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                        detection_writer.writerow([currentClass, conf, frame_timestamp])
                    # Take a snapshot of the frame
                    cv2.imwrite(f'detected_{currentClass}_{frame_timestamp}.jpg', img)
                elif currentClass == 'Hardhat' or currentClass == 'Safety Vest' or currentClass == "Mask":
                    myColor = (0, 255, 0)
                else:
                    myColor = (255, 0, 0)

                cvzone.putTextRect(img, f'{classNames[cls]} {conf}',
                                   (max(0, x1), max(35, y1)), scale=1, thickness=1, colorB=myColor,
                                   colorT=(255, 255, 255), colorR=myColor, offset=5)
                cv2.rectangle(img, (x1, y1), (x2, y2), myColor, 3)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('d'):
        break



from datetime import datetime, timedelta

capture_interval = 5 # in seconds
last_capture_time = datetime.now() - timedelta(seconds=capture_interval)

while True:
    success, img = cap.read()
    results = model(img, stream=True)
    
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
            w, h = x2 - x1, y2 - y1
            # cvzone.cornerRect(img, (x1, y1, w, h))

            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            # Class Name
            cls = int(box.cls[0])
            currentClass = classNames[cls]
            print(currentClass)
            if conf>0.5:
                if currentClass =='NO-Hardhat' or currentClass =='NO-Safety Vest' or currentClass == "NO-Mask":
                    myColor = (0, 0,255)
            
                else:
                    myColor = (255, 0, 0)

                cvzone.putTextRect(img, f'{classNames[cls]} {conf}',
                                   (max(0, x1), max(35, y1)), scale=1, thickness=1,colorB=myColor,
                                   colorT=(255,255,255),colorR=myColor, offset=5)
                cv2.rectangle(img, (x1, y1), (x2, y2), myColor, 3)
                
                # Take snapshot if the specified time interval has passed since the last snapshot
                if datetime.now() - last_capture_time > timedelta(seconds=capture_interval):
                    # Take snapshot and save to file
                    snapshot_filename = f"snapshots/{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                    cv2.imwrite(snapshot_filename, img)
                    
                    # Update last capture time
                    last_capture_time = datetime.now()

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF==ord('d'):
        break
