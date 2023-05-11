from ultralytics import YOLO
import cv2
import cvzone
import math
import datetime
import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage

# cap = cv2.VideoCapture(1)  # For Webcam
# cap.set(3, 1280)
# cap.set(4, 720)
cap = cv2.VideoCapture("../Videos/ppe-3.mp4")  # For Video
# cap=cv2.VideoCapture(0)
model = YOLO("ppe.pt")

classNames = ['Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person', 'Safety Cone',
              'Safety Vest', 'machinery', 'vehicle']

myColor = (0, 0, 255)

# Set up email
smtp_server = "smtp.gmail.com"
smtp_port = 587
smtp_username = "bannuru.kumar@rpsg.in"
smtp_password = "pavaN@555"
sender_email = "pbannuru@gmail.com"
receiver_email = "pavankumarbannuru@gmail.com"


def send_email(subject, body, image_path):
    msg = MIMEMultipart()
    msg['Subject'] = subject
    msg['From'] = sender_email
    msg['To'] = receiver_email

    text = MIMEText(body)
    msg.attach(text)

    with open(image_path, 'rb') as f:
        image_data = f.read()

    image = MIMEImage(image_data, name=os.path.basename(image_path))
    msg.attach(image)

    server = smtplib.SMTP(smtp_server, smtp_port)
    server.starttls()
    server.login(smtp_username, smtp_password)
    server.sendmail(sender_email, receiver_email, msg.as_string())
    server.quit()


snapshot_interval = 30  # in seconds
last_snapshot_time = datetime.datetime.now()

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
            if conf > 0.5:
                if currentClass == 'NO-Hardhat' or currentClass == 'NO-Safety Vest' or currentClass == "NO-Mask":
                    myColor = (0, 0, 255)

                    # Take snapshot if it's time
                    current_time = datetime.datetime.now()
                    time_diff = (current_time - last_snapshot_time).total_seconds()
                    if time_diff > snapshot_interval:
                        snapshot_file_name = f"{currentClass}_{current_time.strftime('%Y%m%d-%H%M%S')}.jpg"
                        cv2.imwrite(snapshot_file_name, img)
                        subject = f"Violation Detected: {currentClass}"
                        body = f"A violation was detected on {current_time.strftime('%Y-%m-%d %H:%M:%S')}. Please check the attached snapshot for details."
                        send_email(subject, body, snapshot_file_name)

                        last_snapshot_time = current_time

                    else:
                        myColor = (0, 255, 0)

                    # Draw Bounding Box and Text
                    cv2.rectangle(img, (x1, y1), (x2, y2), myColor, 2)
                    cv2.putText(img, f"{currentClass} {conf}", (x1 + 10, y1 + 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, myColor, 2)

    cv2.imshow("Output", img)
    if cv2.waitKey(1) == ord('q'):
        break

