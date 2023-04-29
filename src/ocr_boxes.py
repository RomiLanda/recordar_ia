import os
from pathlib import Path
from copy import deepcopy
from itertools import groupby
from more_itertools import windowed, flatten
from pytesseract import image_to_data, Output
from .utils import b64_encoder, save_json, cv2pil
from shapely import box
from statistics import mode

tess_configs = {
    "default": "--psm 11",
    "psm3": "--psm 3",
    "psm4": "--psm 4",
    "psm5": "--psm 5",
    "psm6": "--psm 6",
    "psm12": "--psm 12",
}

TESSERACT_LANG = "spa"
TESSERACT_CONFIG = "default"


def get_token_boxes(image, tesseract_langs: str, tesseract_config: str ) -> list[dict]:
    tess_config = tess_configs.get(tesseract_config, "")
    data = image_to_data(
        image,
        lang=tesseract_langs,
        config=tess_config,
        output_type=Output.DICT,
    )

    data = zip(
        data["text"],
        data["conf"],
        data["left"],
        data["top"],
        data["width"],
        data["height"],
    )
    
    mode_height = mode(data["height"])
    
    # box format =>  (x_left, y_top, x_right, y_bottom)
    token_boxes = map(
        lambda x: {
            "text": x[0],
            "confidence": float(x[1]) / 100,
            "top": x[3],
            "left": x[2],
            "box": (x[2], x[3], x[2] + x[4], x[3] + x[5]),
            "box_polygon": box(x[2], x[3], x[2] + x[4], x[3] + x[5]),
            "box_area": x[4] * x[5],
            "box_height": x[5],
            "x_position": x[2],
            "y_position": x[3],
            "number_presence": True if len(re.findall(r"\d", x[0])) > 0 else False,
            "caps_words_ratio": len(re.findall(r"\b[A-Z][a-z]*", x[0]))/len(x[0].split()) if len(x[0].split()) > 0 else 0.,
            "mode_normalized_height": x[5] / mode_height if mode_height > 0 else 0.,
        },
        data,
    )

    token_boxes = [token for token in token_boxes if token["text"]]

    return token_boxes


MIN_NEW_LINE_OVERLAP = 0.5

def set_line_number(token_boxes: list[dict]) -> list[dict]:
    token_boxes = sorted(
        token_boxes, key=lambda x: ((x["box"][3] + x["box"][1]) / 2)
    )

    token_box_pairs = windowed(token_boxes, 2)
    line = 1
    token_boxes[0]["n_line"] = line
    for prev_token_box, token_box in token_box_pairs:
        prev_box = prev_token_box["box"]
        box = token_box["box"]
        prev_y = (prev_box[3] + prev_box[1]) / 2
        y = (box[3] + box[1]) / 2
        diff = abs(y - prev_y)
        if (box[1] > prev_box[1]) and (box[3] < prev_box[3]):
            token_box["n_line"] = line
            continue

        height = token_box["box_height"]
        if diff >= height * MIN_NEW_LINE_OVERLAP:
            line += 1

        token_box["n_line"] = line

    return token_boxes


def get_line_groups(token_boxes: list[dict]):
    line_groups = groupby(token_boxes, key=lambda x: x["n_line"])
    line_groups = map(
        lambda x: sorted(x[1], key=lambda x: x["box"][0]), line_groups
    )

    return line_groups


def set_token_box_ids(
    token_boxes: list[dict[str]],
    image_id: str,
) -> list[dict[str]]:

    line_groups = get_line_groups(token_boxes)
    sorted_token_boxes = flatten(line_groups)
    token_boxes_ = []
    for idx, token_box in enumerate(sorted_token_boxes, start=1):
        token_id = f"{image_id}-{idx}"
        token_id = b64_encoder(token_id)
        token_box_ = {"id": token_id, **token_box, "n_token": idx}
        token_boxes_.append(token_box_)

    return token_boxes_


def apply_tesseract(
    data_item,
    tesseract_langs: str = "spa",
    tesseract_config: str = "default",
    output_path: str = "",
):

    data_item = deepcopy(data_item)
    image_path = data_item["file_path"]
    img = data_item["img_bitmap"]
    image = cv2pil(img)

    image_path = data_item["file_path"]
    filename = os.path.basename(image_path)
    image_id = f"{filename}"

    token_boxes = get_token_boxes(image, tesseract_langs, tesseract_config)
    token_boxes = set_line_number(token_boxes)
    token_boxes = set_token_box_ids(
        token_boxes,
        image_id,
    )

    if not token_boxes:
        logger.warning(f"WARNING no boxes for image => {image_path}")
        return

    data_item["token_boxes"] = token_boxes

    if output_path:
        path = Path(output_path)
        path.mkdir(parents=True, exist_ok=True)
        file_hash = b64_encoder(data_item["file_path"])
        # save_json(token_boxes, f"{output_path}/{file_hash}.json") TODO Polygon not serializable

    return data_item