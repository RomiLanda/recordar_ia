import os
import cv2
import numpy as np
from PIL import ImageDraw
import matplotlib.pyplot as plt 
from PIL import ImageFont

from .utils import cv2pil, get_boxes_line, get_line_center

FONT = ImageFont.truetype(
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 10
)

def doc_debug(data_item, out_path):
    filename = os.path.basename(data_item["file_path"])
    out_file_path = f"{out_path}/{filename}"

    segments = data_item["segments"]

    image = cv2pil(data_item["img_bitmap"])
    token_boxes = data_item["token_boxes"]

    boxes_map = {
        token_box["id"]: token_box["box"] for token_box in token_boxes
    }

    draw = ImageDraw.Draw(image)
    for token_box in token_boxes:
        box = token_box["box"]
        n_line = token_box["n_line"]
        label = token_box["label"]

        color = "red" if label != -1 else "black"
        draw.rectangle(box, outline=color, width=1)
        draw.text(
            (box[0], box[1]),
            f"{n_line}:{label}",
            font=FONT,
            anchor="rd",
            align="left",
            fill=color,
        )

    for segment in segments:
        box = segment["box"]
        draw.rectangle(box, outline="blue", width=1)

    for src, tgt, attr in data_item["doc_graph"].edges(data=True):
        src_box = boxes_map[src]
        tgt_box = boxes_map[tgt]

        boxes_line = get_boxes_line(src_box, tgt_box)
        draw.line(boxes_line, fill="green", width=1)
        x, y = get_line_center(boxes_line)
        draw.text(
            (x, y),
            str(round(attr["weight"], 3)),
            font=FONT,
            anchor="rd",
            align="left",
            fill="blue",
        )

    image.save(out_file_path)


