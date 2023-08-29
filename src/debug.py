import os
from PIL import ImageFont
from PIL import ImageDraw

from .utils import cv2pil, get_boxes_line, get_line_center

OCR_DEBUG = True    # if False, turns into Label debug output images

FONT = ImageFont.truetype(
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 14
)

def doc_debug(data_item, out_path: str, train_flow: bool):
    filename = os.path.basename(data_item["file_path"])
    out_file_path = f"{out_path}/{filename}"

    image = cv2pil(data_item["img_bitmap"])
    token_boxes = data_item["token_boxes"]

    boxes_map = {
        token_box["id_line_group"]: token_box["box"] for token_box in token_boxes
    }

    draw = ImageDraw.Draw(image)
    for token_box in token_boxes:
        box = token_box["box"]
        n_line = token_box["id_line_group"]
        if train_flow:
            text_draw = token_box["text"] if OCR_DEBUG else f'{n_line}:{token_box["label"]}'
            align_text = 'center' if OCR_DEBUG else 'left'
            anchor_text = 'lt' if OCR_DEBUG else 'rd'
            color = "red" if text_draw != "Indefinido" else "black"
            draw.rectangle(box, outline=color, width=1)
            draw.text(
                (box[0], box[1]),
                f"{text_draw}",
                font=FONT,
                anchor=anchor_text,
                align=align_text,
                fill=color,
            )
        else:
            color = "black"
            draw.rectangle(box, outline=color, width=1)
            draw.text(
                (box[0], box[1]),
                f"{n_line}",
                font=FONT,
                anchor="rd",
                align="left",
                fill=color,
            )           

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

    if train_flow:
        segments = data_item["segments"]
        for segment in segments:
            box = segment["box"]
            draw.rectangle(box, outline="blue", width=1)

    image.save(out_file_path)


