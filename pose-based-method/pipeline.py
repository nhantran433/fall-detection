import sys
sys.path.append('src/')

from PoseEstimator import PoseEstimator
from FallDetector import FallDetector
from RulebaseChecker import RulebaseChecker
from YoloDetector import YoloDetector

import cv2

class Pipeline:
    def __init__(self):
        self.pose_estimator = PoseEstimator()
        self.fall_detector = FallDetector()
        self.rulebase_checker = RulebaseChecker()
        self.yolo_detector = YoloDetector()

    def run(self, input_path, output_path):
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise IOError('Cannot open video')

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Detect persons and furniture
            persons_detections, furnitures_detections = self.yolo_detector.detect(frame)
            furnitures = [furniture[0] for furniture in persons_detections]
            print("Detected persons:", persons_detections)
            print("Detected furniture:", furnitures_detections)

            # Pose estimation
            keypoints, boxes = self.pose_estimator.detect(frame)  # Return keypoints and boxes of people (multiple people)
            print("Keypoints:", keypoints)
            print("Boxes:", boxes)

            if len(boxes) == 0:
                out.write(frame)
                continue
            for keypoint, box in zip(keypoints, boxes):
                # Pre-check and prediction
                pre_check = self.fall_detector.pre_check(keypoint, box)  # Only one person each time
                prediction = self.fall_detector.predict(keypoint)
                print("Pre-check:", pre_check)
                print("Prediction:", prediction)

                # Check if fall is detected and draw accordingly
                if prediction and pre_check:
                    if len(boxes) > 1 or self.rulebase_checker.is_on_furniture(box, furnitures):
                        if self.rulebase_checker.is_on_furniture(box, furnitures):
                            for furniture in furnitures_detections:
                                iou = self.rulebase_checker.calculate_iou(box, furniture[0])
                                if iou > 0.7:
                                    a, b, c, d = map(int, furniture)
                                    cv2.rectangle(frame, (a, b), (c, d), (255, 0, 0), 2)
                                    cv2.putText(frame, f'{furniture[1]}', (a, b - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                                    print(f"Drawing blue box at: {(a, b, c, d)}")
                        x1, y1, x2, y2 = map(int, box)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(frame, "safe", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        cv2.putText(frame, "fall", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        print(f"Drawing red box at: {(x1, y1, x2, y2)}")
                    else:
                        x1, y1, x2, y2 = map(int, box)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(frame, "not safe", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        cv2.putText(frame, "fall", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        print(f"Drawing red box at: {(x1, y1, x2, y2)}")
                else:
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, "safe", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(frame, "not fall", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    print(f"Drawing green box at: {(x1, y1, x2, y2)}")
            out.write(frame)
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print(fps)

    def run_image(self, image_path, output_path):
        frame = cv2.imread(image_path)
        persons_detections, furnitures_detections = self.yolo_detector.detect(frame)
        keypoints, boxes = self.pose_estimator.detect(frame)  # Return keypoints and boxes of people (multiple people)

        if len(boxes) == 0:
            print("No person detected")
            return
        for keypoint, box in zip(keypoints, boxes):
            # Pre-check and prediction
            pre_check = self.fall_detector.pre_check(keypoint, box)  # Only one person each time
            prediction = self.fall_detector.predict(keypoint)
            print("Pre-check:", pre_check)
            print("Prediction:", prediction)

            # Check if fall is detected and draw accordingly
            if prediction and pre_check:
                if len(boxes) > 1 or self.rulebase_checker.is_on_furniture(box, furnitures_detections):
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, "fall", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    print(f"Drawing red box at: {(x1, y1, x2, y2)}")
            else:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, "not fall", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                print(f"Drawing green box at: {(x1, y1, x2, y2)}")

        # Save the frame with annotations
        cv2.imwrite(output_path, frame)
        print(f"Image saved at {output_path}")

if __name__ == '__main__':

    # Load the video
    input_path = r"C:\Users\NHAN\fall-detection\new-code\data\input\video_demo6.mp4"
    output_path = r'.\data\output\output_demo6.mp4'

    pipeline = Pipeline()
    pipeline.run(input_path, output_path)

    