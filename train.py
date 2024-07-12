from ultralytics import YOLOv10
from roboflow import Roboflow

model = YOLOv10("/home/via/weights/yolov10s.pt")

#if you want to train own custom model you need to make a datasets folder and put your dataset in it
model.train(data = "/home/via/datasets/CARLA-Object-Detection-2/data.yaml", epochs = 65, imgsz=800, plots=True)
