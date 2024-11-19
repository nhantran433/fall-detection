import cv2
from ultralytics import YOLO

class YoloDetector:
    def __init__(self):
        self.model = YOLO('../checkpoints/yolov8n.pt')

    def detect(self, image):
        results = self.model(image, verbose = False)
        person_detections = []
        furnitures_detections = []
        # dict_yolo = {}

        for result in results:
            boxes = result.boxes.xyxy.tolist()
            classes = result.boxes.cls.tolist()
            names = result.names
            confidences = result.boxes.conf.tolist()

            for box, cls, conf in zip(boxes, classes, confidences):
                x1, y1, x2, y2 = box
                w, h = x2 - x1, y2 - y1
                confidence = conf
                detected_class = cls
                name = names[int(cls)]
                if confidence > 0.5 and name == 'person':
                    # tuple_people = (x1, y1, x2, y2)
                    # dict_yolo[tuple_people] = 0
                    person_detections.append(([x1, y1, w, h], confidence, detected_class))

                if confidence > 0.2 and name == 'couch' or name == 'bed':
                    furnitures_detections.append(([x1, y1, x2, y2], name))

        return person_detections, furnitures_detections