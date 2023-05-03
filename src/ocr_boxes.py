import os
from pathlib import Path
from copy import deepcopy
from itertools import groupby
from more_itertools import windowed, flatten
from pytesseract import image_to_data, Output
from .utils import b64_encoder, save_json, cv2pil
from shapely import box

tess_configs = {
    "default": "--psm 11",
    "psm3": "--psm 3",
    "psm4": "--psm 4",
    "psm5": "--psm 5",
    "psm6": "--psm 6",
    "psm12": "--psm 12",
    "with_whitelist": r'-c tessedit_char_whitelist="AÁBCDEÉFGHIÍJKLMNÑOÓPQRSTUÚVWXYZaábcdeéfghiíjklmnñoópqrstuúvwxyz0123456789 -_/.,:;()"',
}

TESSERACT_LANG = "spa"
TESSERACT_CONFIG = "with_whitelist"


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
        },
        data,
    )

    token_boxes = [token for token in token_boxes if token["text"]]

    return token_boxes


def blank_filter(df_data):
    """
    Returns dataframe without elements with text NaN or empty from pytesseract data as dataframe
    """   
    mask_not_blank = (df_data['text'].str.strip() != '')
    df_data = df_data[mask_not_blank].dropna(subset=['text'])
    return df_data


def get_line_group_token_boxes(image, tesseract_langs: str, tesseract_config: str ) -> list[dict]:
    tess_config = tess_configs.get(tesseract_config, "")
    df_data = image_to_data(
        image,
        lang=tesseract_langs,
        config=tess_config,
        output_type=Output.DATAFRAME,
    )

    df_data = blank_filter(df_data)

    df_data['x2'] = df_data['left'] + df_data['width']
    df_data['y2'] = df_data['top'] + df_data['height']
    df_data['id_line_group'] = df_data.apply(lambda row:
                                                 'id_' + str(row['block_num']) +
                                                 str(row['par_num']) +
                                                 str(row['line_num']), axis=1
                                            )
    line_groups_ids = df_data['id_line_group'].value_counts().keys().to_list()

    groups_boundaries = {}
    for g in line_groups_ids:
        group_boundaries = {'x1': 0, 'y1': 0, 'x2': 0, 'y2': 0}
        df_g = df_data[df_data.id_line_group == g]
        group_boundaries['x1'] = df_g['left'].min()
        group_boundaries['y1'] = df_g['top'].min()
        group_boundaries['x2'] = df_g['x2'].max()
        group_boundaries['y2'] = df_g['y2'].max()
        group_boundaries['text'] = " ".join(df_g['text'])
        groups_boundaries[g] = group_boundaries

    token_line_groups_boxes = map(
        lambda x: {
            "top": groups_boundaries[x]['y1'],
            "left": groups_boundaries[x]['x1'],
            "box": (groups_boundaries[x]['x1'],
                    groups_boundaries[x]['y1'],
                    groups_boundaries[x]['x2'],
                    groups_boundaries[x]['y2']),
            "box_polygon": box(groups_boundaries[x]['x1'],
                               groups_boundaries[x]['y1'],
                               groups_boundaries[x]['x2'],
                               groups_boundaries[x]['y2']),
            "box_area": groups_boundaries[x]['x1'] * groups_boundaries[x]['y1'],
            "box_height": groups_boundaries[x]['y2'] - groups_boundaries[x]['y1'],
            "box_width" : groups_boundaries[x]['x2'] - groups_boundaries[x]['x1'],
            "x_position": groups_boundaries[x]['x1'],
            "y_position": groups_boundaries[x]['y1'],
            "text": groups_boundaries[x]['text']
            
        },
            groups_boundaries
    )

    return token_line_groups_boxes


MIN_NEW_LINE_OVERLAP = 0.15

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

    token_boxes = get_line_group_token_boxes(image, tesseract_langs, tesseract_config)
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