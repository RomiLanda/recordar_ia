import cv2
import json
import numpy as np
from PIL import Image
from copy import deepcopy
from base64 import b64encode, b64decode


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


def image_unload(data_item):
    """Release image memory
    Returns:
        DataItem: data without loaded image
    """
    data_item = deepcopy(data_item)
    data_item.pop("img_bitmap", None)

    return data_item


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