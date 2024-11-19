from tensorflow.keras.models import load_model
import math
import numpy as np

class FallDetector:
    def __init__(self):
        self.model = load_model('./checkpoints/model_cnn_lstm.keras')

    def calculate_len_factor(self, shoulder_x, shoulder_y, hip_x, hip_y):
        return math.sqrt((shoulder_y - hip_y) ** 2 + (shoulder_x - hip_x) ** 2)

    def pre_check(self, keypoints, boxes):
        # Extract bounding box coordinates
        xmin, ymin, xmax, ymax = boxes

        # Extract keypoints
        left_shoulder_x, left_shoulder_y = keypoints[0], keypoints[1]
        right_shoulder_x, right_shoulder_y = keypoints[2], keypoints[3]
        left_hip_x, left_hip_y = keypoints[12], keypoints[13]
        right_hip_x, right_hip_y = keypoints[14], keypoints[15]
        left_ankle_x, left_ankle_y = keypoints[20], keypoints[21]
        right_ankle_x, right_ankle_y = keypoints[22], keypoints[23]

        is_left_falling = False
        is_right_falling = False

        # Calculate length factors and check for left side falling condition
        if all([left_shoulder_x, left_shoulder_y, left_hip_x, left_hip_y, left_ankle_x, left_ankle_y]):
            len_factor_left = self.calculate_len_factor(left_shoulder_x, left_shoulder_y, left_hip_x, left_hip_y)
            is_left_falling = (left_shoulder_y > left_ankle_y - len_factor_left and
                            left_hip_y > left_ankle_y - (len_factor_left / 2) and
                            left_shoulder_y > left_hip_y - (len_factor_left / 2))

        # Calculate length factors and check for right side falling condition
        if all([right_shoulder_x, right_shoulder_y, right_hip_x, right_hip_y, right_ankle_x, right_ankle_y]):
            len_factor_right = self.calculate_len_factor(right_shoulder_x, right_shoulder_y, right_hip_x, right_hip_y)
            is_right_falling = (right_shoulder_y > right_ankle_y - len_factor_right and
                                right_hip_y > right_ankle_y - (len_factor_right / 2) and
                                right_shoulder_y > right_hip_y - (len_factor_right / 2))

        # Calculate bounding box differences
        dx = xmax - xmin
        dy = ymax - ymin
        difference = dy - dx

        # if is_left_falling or is_right_falling or difference < -30:
        if difference < -10:
            return True

        return False

    def predict(self, keypoints_extracted = None):
        if keypoints_extracted is None:
            return False
        else:
            data = np.array(keypoints_extracted)
        if data is None or len(data) == 0 :
            return False
        data = data[10:]
        prediction = False
        if np.all([data[1], data[13], data[17]]) or np.all([data[1], data[13], data[21]]) or np.all([data[3], data[15], data[19]]) or np.all([data[3], data[15], data[23]]):
            data = data.reshape(1, data.shape[0], 1)
            model_pred = self.model.predict(data,verbose = 0).reshape(-1,)
            if model_pred[-1] <= 0.5:
                prediction = True

        return prediction


