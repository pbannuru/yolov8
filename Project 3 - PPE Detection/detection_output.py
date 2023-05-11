from ultralytics import YOLO
import cv2
import cvzone
import math
import os
from datetime import datetime, timedelta
# Create output directory if it doesn't exist
output_dir = "output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# cap = cv2.VideoCapture(0)  # For Webcam
# cap.set(3, 1280)
# cap.set(4, 720)
# cap = cv2.VideoCapture("../Videos/ppe-3.mp4")  # For Video
cap=cv2.VideoCapture(0)
model = YOLO("ppe.pt")

classNames = ['Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person', 'Safety Cone',
              'Safety Vest', 'machinery', 'vehicle']


capture_interval = 5 # in seconds
last_capture_time = datetime.now() - timedelta(seconds=capture_interval)


myColor = (0, 0, 255)
while True:
    success, img = cap.read()
    results = model(img, stream=True)
    
    # List to store detection data
    detection_data = []
    
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
            
            # Add detection data to list
            detection_data.append({"class": currentClass, "confidence": conf, "box": (x1, y1, x2, y2)})
            
            if conf > 0.5:
                if currentClass == 'NO-Hardhat' or currentClass == 'NO-Safety Vest' or currentClass == "NO-Mask":
                    myColor = (0, 0, 255)
                    
                    # Save snapshot of object without proper PPE
                    snapshot_filename = os.path.join(output_dir, f"{currentClass}_{conf}.jpg")
                    cv2.imwrite(snapshot_filename, img[y1:y2, x1:x2])
                    
                # elif currentClass == 'Hardhat' or currentClass == 'Safety Vest' or currentClass == "Mask":
                #     myColor = (0, 255, 0)
                else:
                    myColor = (255, 0, 0)

                cvzone.putTextRect(img, f'{classNames[cls]} {conf}',
                                   (max(0, x1), max(35, y1)), scale=1, thickness=1,colorB=myColor,
                                   colorT=(255, 255, 255),colorR=myColor, offset=5)
                cv2.rectangle(img, (x1, y1), (x2, y2), myColor, 3)

    cv2.imshow("Image", img)
    
    # Save detection data to disk
    detection_filename = os.path.join(output_dir, "detection_data.txt")
    with open(detection_filename, "w") as f:
        f.write(str(detection_data))

    # Exit loop when 'q' key is pressed
    if cv2.waitKey(1) == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
 


