import numpy as np
import cv2
from typing import List, Union


def get_random_color(seed):
    gen = np.random.default_rng(seed)
    color = tuple(gen.choice(range(256), size=3))
    color = tuple([int(c) for c in color])
    return color


def draw_detections(img: np.array, bboxes: List[List[int]], classes: List[int],
                    class_labels: Union[List[str], None]):
    for bbox, cls in zip(bboxes, classes):
        x1, y1, x2, y2 = bbox

        color = get_random_color(int(cls))
        img = cv2.rectangle(
            img, (int(x1), int(y1)), (int(x2), int(y2)), color, 3
        )
        
        if class_labels:
            label = class_labels[int(cls)]

            x_text = int(x1)
            y_text = max(15, int(y1 - 10))
            img = cv2.putText(
                img, label, (x_text, y_text), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, color, 1, cv2.LINE_AA
            )

    return img
