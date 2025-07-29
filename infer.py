from ultralytics import YOLO
import cv2

model = YOLO("yolo11m_safety.pt")
img= r"test.jpeg"
results = model(img)
predicted = results[0].plot()

cv2.imwrite('predicted.jpg', predicted)