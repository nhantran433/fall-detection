import numpy as np
from ultralytics import YOLO

class PoseEstimator:
    def __init__(self):
        self.model = YOLO('../checkpoints/yolov8n-pose.pt')

    def detect(self, image):
        data = []
        boxes = []
        results = self.model(image,verbose = False)
        for result in results:
            if len(result.boxes) == 1:
                result_keypoints = result.keypoints.xyn.cpu().numpy()
                for keypoint in result_keypoints:
                    lst_keypoints = [coord for idx in range(17) for coord in keypoint[idx][:2]]
                    boxes.append(result.boxes.xyxy[0].tolist())
                data.append(lst_keypoints)
            elif len(result.boxes) > 1:
                result_keypoints =  result.keypoints.xyn.cpu().numpy()
                lst_much_keypoints = []
                for keypoint in result_keypoints:
                    lst_keypoints = [coord for idx in range(17) for coord in keypoint[idx][:2]]
                    lst_much_keypoints.append(lst_keypoints)
                for box in result.boxes.xyxy:
                    boxes.append(box.tolist())
                data = lst_much_keypoints

        return data, boxes