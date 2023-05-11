from ultralytics import YOLO
import cv2

#model = YOLO('../Yolo-Weights/yolov8l.pt')
model = YOLO('../Yolo-Weights/yolov8n.pt')
results = model("Images/1.png", show=True)
cv2.waitKey(0)

