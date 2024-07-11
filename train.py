from ultralytics import YOLO


from roboflow import Roboflow



rf = Roboflow(api_key="AW8ywqbBjeRFvAXEVXIE")
project = rf.workspace("mc-rb0d2").project("carla-cars-uqjfy")
version = project.version(10)
dataset = version.download("folder")


model = YOLO("yolov8n-cls.pt")

if __name__ == '__main__':
    model.train(data="CARLA-CARS-10",epochs=75)
