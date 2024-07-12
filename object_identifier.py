import numpy as np
import pygame
from ultralytics import YOLOv10
import cv2
import io
import torch
import os


class ObjectIdentifier(object):
    torch.cuda.set_device(0)
    model = YOLOv10("runs/detect/train3/weights/best.pt")
    img_count = 0
    class_colors = {
            "Car" : (202, 214, 124),
            "Traffic Light" : (71, 233, 255),
            "Bench" : (26, 188, 20),
            "Bicycle" : (112, 164, 209),
            "Motorcycle" : (196, 132, 54),
            "Person" : (186, 22, 137),
            "Plant" : (163, 106, 148),
            "Sign" : (76, 35, 119),
            "Street Light" : (65, 189, 211),
            "Street Label" : (209, 25, 114),
            "Trash Can" : (126, 209, 117),
            "Crosswalk" : (158, 74, 84),
            "city-objects" : (20, 7, 15)
            
    }

    @staticmethod
    def object_identifier(display):
        view = pygame.surfarray.array3d(display).transpose([1, 0, 2])
        img = cv2.cvtColor(view, cv2.COLOR_RGB2BGR)
        
        color = (202, 214, 124)

        result = ObjectIdentifier.model(img)
        for i, box in enumerate(result[0].boxes.xyxy):
            x1, y1, x2, y2 = map(int, box[:4])
            class_name = ObjectIdentifier.model.names[int(result[0].boxes.cls[i])]
            confidence_score = float(result[0].boxes.conf[i])
            color = ObjectIdentifier.class_colors[class_name]
            ObjectIdentifier.draw_box(x1, y1, x2, y2, display, color)
            font_name = "fonts/calibri.ttf"
            
            font = pygame.font.Font(font_name, 20)
            
            text_surface = font.render(f"{class_name} {round(confidence_score, 2)}", True, (255, 255, 255))

            text_width, text_height = text_surface.get_size()
            highlight_rect = pygame.Rect(x1, y1 - 20, text_width + 2, text_height + 2)

            pygame.draw.rect(display, color, highlight_rect)

            display.blit( text_surface, (x1, y1-20))

    def draw_box(x1, y1, x2, y2, display, color):
        pygame.draw.rect(display, color, (x1, y1, x2-x1, y2-y1), 2)
    