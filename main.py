from ultralytics import YOLO
import cv2

#model = YOLO('../Yolo-Weights/yolov8l.pt')
# results = model("Images/3.png", show=True)
cv2.waitKey(0)
cap=cv2.VideoCapture(0)
# while True:
#     isTrue,frames=cap.read()
#     cv2.imshow('video',frames)
#     if cv2.waitKey(1) & 0xFF==ord('d'):
#         break

model = YOLO("ppe.pt")

classNames = ['Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person', 'Safety Cone',
              'Safety Vest', 'machinery', 'vehicle']
myColor = (0, 0, 255)
while True:
    success, img = cap.read()
    results = model(img, stream=True)
    for r in results:
        print(r)

    cv2.imshow("Image", img)
    #cv2.waitKey(1)
    if cv2.waitKey(1) & 0xFF==ord('d'):
        break