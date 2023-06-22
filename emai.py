from ultralytics import YOLO
import cv2
import cvzone
import math
import time
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from email.utils import COMMASPACE
from email import encoders

# cap = cv2.VideoCapture(1)  # For Webcam
# cap.set(3, 1280)
# cap.set(4, 720)
cap = cv2.VideoCapture("../Videos/ppe-3.mp4")  # For Video
# cap=cv2.VideoCapture(0)
model = YOLO("ppe.pt")


classNames = ['Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person', 'Safety Cone',
              'Safety Vest', 'machinery', 'vehicle']
myColor = (0, 0, 255)

# Email settings
send_from = 'sender@example.com'
send_to = 'recipient@example.com'
subject = 'PPE Detection Snapshot'
smtp_server = 'smtp.example.com'
smtp_port = 587
smtp_username = 'username'
smtp_password = 'password'


def send_email(send_from, send_to, subject, message, files=[], server="localhost", port=587, username='', password='', use_tls=True):
    msg = MIMEMultipart()
    msg['From'] = send_from
    msg['To'] = COMMASPACE.join(send_to) if isinstance(send_to, list) else send_to
    msg['Subject'] = subject

    msg.attach(MIMEText(message))

    for f in files:
        part = MIMEBase('application', "octet-stream")
        part.set_payload(open(f, "rb").read())
        encoders.encode_base64(part)
        part.add_header('Content-Disposition', 'attachment', filename=f)
        msg.attach(part)

    smtp = smtplib.SMTP(server, port)
    if use_tls:
        smtp.starttls()
    smtp.login(username, password)
    smtp.sendmail(send_from, send_to, msg.as_string())
    smtp.quit()


# Snapshot settings
snapshot_interval = 60  # Snapshot every 60 seconds
last_snapshot_time = time.time()

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
                    if time.time() - last_snapshot_time >= snapshot_interval:
                        snapshot_filename = f"{currentClass}_{time.strftime('%Y%m%d


from ultralytics import YOLO
import cv2
import cvzone
import math
import time
import smtplib
import os.path
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.image import MIMEImage
from email.utils import COMMASPACE
from email import encoders


# Function to send email with attached image files
def send_email(subject, message, from_email, to_email, file_paths, smtp_server, smtp_port, smtp_username, smtp_password):
    msg = MIMEMultipart()
    msg['From'] = from_email
    msg['To'] = COMMASPACE.join(to_email)
    msg['Subject'] = subject
    msg.attach(MIMEImage(file(file_paths[0], 'rb').read()))
    msg.attach(MIMEImage(file(file_paths[1], 'rb').read()))
    msg.attach(MIMEImage(file(file_paths[2], 'rb').read()))
    msg.attach(MIMEImage(file(file_paths[3], 'rb').read()))

    smtp = smtplib.SMTP(smtp_server, smtp_port)
    smtp.starttls()
    smtp.login(smtp_username, smtp_password)
    smtp.sendmail(from_email, to_email, msg.as_string())
    smtp.quit()


cap = cv2.VideoCapture("../Videos/ppe-3.mp4")  # For Video
model = YOLO("ppe.pt")

classNames = ['Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person', 'Safety Cone',
              'Safety Vest', 'machinery', 'vehicle']
myColor = (0, 0, 255)

prev_time = 0
time_interval = 60 # in seconds

snapshot_counter = 0
snapshot_paths = []

while True:
    success, img = cap.read()
    results = model(img, stream=True)

    current_time = time.time()
    elapsed_time = current_time - prev_time

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
                    if elapsed_time >= time_interval:
                        snapshot_path = f"snapshot_{snapshot_counter}.jpg"
                        cv2.imwrite(snapshot_path, img)
                        snapshot_paths.append(snapshot_path)
                        snapshot_counter += 1
                elif currentClass == 'Hardhat' or currentClass == 'Safety Vest' or currentClass == "Mask":
                    myColor = (0, 255, 0)
                else:
                    myColor = (255, 0, 0)

                cvzone.putTextRect(img, f'{classNames[cls]} {conf}',
                                   (max(0, x1), max(35, y1)), scale
