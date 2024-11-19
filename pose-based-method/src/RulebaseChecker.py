

class RulebaseChecker:
    def __init__(self):
        pass

    def calculate_iou(self, box1, box2):
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2

        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)

        inter_width = max(0, xi2 - xi1)
        inter_height = max(0, yi2 - yi1)
        inter_area = inter_width * inter_height

        box1_area = w1 * h1
        box2_area = w2 * h2

        union_area = box1_area + box2_area - inter_area

        iou = inter_area / union_area if union_area != 0 else 0
        return iou
    
    def is_on_furniture(self, person_bb, furnitures_bb):
        for furniture in furnitures_bb:
            iou = self.calculate_iou(person_bb, furniture)
            if iou > 0.7:
                return True
        return False

    def is_alone(person_detections):
        if len(person_detections) > 1:
            return False
