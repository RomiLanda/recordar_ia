import cv2
import json
import numpy as np
from PIL import Image
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