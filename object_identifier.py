import numpy as np
import pygame
from ultralytics import YOLO
import cv2
import io
import torch
import os

class ObjectIdentifier(object):
    torch.cuda.set_device(0)
    identify_model = YOLO("yolov8n.pt")
    classify_model = YOLO("runs/classify/train/weights/best.pt")
    img_count = 0
    class_colors = {
            "Car" : (202, 214, 124),
            "Traffic Light" : (71, 233, 255),
            "Bench" : (26, 188, 20),
            "Bicycle" : (112, 164, 209),
            "Motorcycle" : (196, 132, 54),
            "Pole" : (186, 22, 137),
            "Plant" : (163, 106, 148),
            "Sign" : (76, 35, 119),
            "Street Object" : (65, 189, 211),
            "Traffic Object" : (209, 25, 114),
            "Trash Can" : (126, 209, 117),
            "Truck" : (158, 74, 84),
            "Unlabeled" : (20, 7, 15)
            
    }

    @staticmethod
    def object_identifier(display):
        view = pygame.surfarray.array3d(display).transpose([1, 0, 2])
        img = cv2.cvtColor(view, cv2.COLOR_RGB2BGR)
        font = pygame.font.Font("fonts/calibri.ttf", 20)
        color = (202, 214, 124)

        result = ObjectIdentifier.identify_model(img)
        for coords in result[0].boxes.xyxy:
            x1, y1, x2, y2 = map(int, coords[:4])

            pygame.draw.rect(display, color, (x1, y1, x2-x1, y2-y1), 2)
            rect = pygame.Rect(x1, y1, x2-x1, y2-y1)
            sub = display.subsurface(rect)
            path = f"imgs/object_{ObjectIdentifier.img_count}.png"
            pygame.image.save(sub, path)
            result = ObjectIdentifier.classify_model.predict(path)[0]

            class_name = result.names[result.probs.top1]
            score = round(float(result.probs.top1conf.cpu().numpy()), 2)
            color = ObjectIdentifier.class_colors[class_name]


            text_surface = font.render(f"{class_name} {score}", True, (255, 255, 255))
            display.blit(text_surface, (x1, y1-20))
            os.remove(path)
            ObjectIdentifier.img_count += 1

