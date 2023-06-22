from ultralytics import YOLO
import cv2
import cvzone
import math
import os
from datetime import datetime

# Create output directory if it doesn't exist
output_dir = "output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# OpenCV VideoCapture object
#cap = cv2.VideoCapture("../Videos/ppe-3.mp4")  # For Video
cap=cv2.VideoCapture(0)

model = YOLO("ppe.pt")

# List of class names
classNames = ['Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person', 'Safety Cone',
              'Safety Vest', 'machinery', 'vehicle']

# Color for bounding boxes and text
myColor = (0, 0, 255)
# Loop through video frames
while True:
    # Read frame from video capture
    success, img = cap.read()
    results = model(img, stream=True)
    
    # List to store detection data
    detection_data = []
    
    for r in results:
        boxes= r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
            w, h = x2 - x1, y2 - y1
    
        # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            # Class Name
            cls = int(box.cls[0])
            currentClass = classNames[cls]
            print(currentClass)
    
            # Class name
            currentClass = classNames[cls]
            
            # Add detection data to list
            detection_data.append({"class": currentClass, "confidence": conf, "box": box})
        
        # Draw bounding box and text on image
            if conf > 0.5:
                if currentClass in ('NO-Hardhat', 'NO-Safety Vest', 'NO-Mask'):
                    myColor = (0, 0, 255)
                    
                    # Save snapshot of object without proper PPE
                    snapshot_filename = os.path.join(output_dir, f"{currentClass}_{conf:.2f}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.jpg")
                    cv2.imwrite(snapshot_filename, img[y1:y2, x1:x2])
                    
                else:
                    myColor = (0, 255, 0)
                    
                # Draw bounding box and text
                cv2.rectangle(img, (x1, y1), (x2, y2), myColor, 3)
                cvzone.putTextRect(img, f'{currentClass} {conf:.2f}', (max(0, x1), max(35, y1)), scale=1, thickness=1, colorB=myColor, colorT=(255, 255, 255), colorR=myColor, offset=5)
        
    # Display image
    cv2.imshow("Image", img)
    
    # Save detection data to disk
    detection_filename = os.path.join(output_dir, f"detection_data_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt")
    with open(detection_filename, "w") as f:
        f.write(str(detection_data))
    
    # Exit loop when 'q' key is pressed
    if cv2.waitKey(1) == ord('q'):
        break

# # Release resources
# cap.release()
# cv2.destroyAllWindows()