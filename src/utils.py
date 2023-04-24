import cv2
import json
import numpy as np
from PIL import Image
from copy import deepcopy
from base64 import b64encode, b64decode
from shapely.geometry import box as shapely_box


def b64_encoder(x: str) -> str:
    encoded = b64encode(x.encode()).decode()
    return encoded


def save_json(json_data, file_path: str):
    with open(file_path, "w") as f:
        f.write(json.dumps(json_data, indent=4, ensure_ascii=False))
        

def cv2pil(cv_image: np.ndarray) -> Image:
    image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image)
    return pil_image

# -------- LABELS------------

def get_label_token(polygon_token, data_item):
    label_candidates = {'etiqueta': -1, 'perc': -1, 'label': -1}
    i=0
    for segmento in data_item['segments']:
        if polygon_token.intersects(segmento['polygon']):
            i+=1
            perc = polygon_token.intersection(segmento['polygon']).area
            if (label_candidates['perc'] < perc):
                label_candidates = {'label': segmento['label'], 'perc': perc}
    
    return label_candidates['label']


def get_label_tokens(data_item):
    for token_box in data_item['token_boxes']:
        token_box['label'] = get_label_token(token_box['box_polygon'],data_item)
    return data_item


# -------- GEOMETRICS------------

def get_boxes_line(box_1, box_2):
    #  box => (x1, y1, x2, y2)
    x1 = (box_1[0] + box_1[2]) / 2
    y1 = (box_1[1] + box_1[3]) / 2

    x2 = (box_2[0] + box_2[2]) / 2
    y2 = (box_2[1] + box_2[3]) / 2

    line = (x1, y1, x2, y2)
    return line


def get_line_center(line):
    x = int(line[2] - (line[2] - line[0]) / 2)
    y = int(line[3] - (line[3] - line[1]) / 2)
    return x, y


def get_boxes_ditance(box_1, box_2) -> float:
    box_1 = shapely_box(*box_1)
    box_2 = shapely_box(*box_2)
    ditance = box_1.distance(box_2)
    return ditance

    