import cv2
from shapely import box
from copy import deepcopy
from pytesseract import image_to_data, Output
from .utils import cv2pil, blank_filter, vertical_filter, phantom_ocr_filter, symbols_string_filter
from .get_photos import add_photo_token_boxes, get_photo_polygons


tess_configs = {
    "default": "--psm 11",
    "psm3": "--psm 3",
    "psm4": "--psm 4",
    "psm5": "--psm 5",
    "psm6": "--psm 6",
    "psm12": "--psm 12",
    "with_whitelist": r'-c tessedit_char_whitelist="AÁBCDEÉFGHIÍJKLMNÑOÓPQRSTUÚVWXYZaábcdeéfghiíjklmnñoópqrstuúvwxyz0123456789 -_/.,:;()"',
}

BUFFER_RATIO = 1.8

def tesseract_word_boxes(image, tesseract_langs: str, tesseract_config: str):
    tess_config = tess_configs.get(tesseract_config, "")
    df_data = image_to_data(
        image,
        lang=tesseract_langs,
        config=tess_config,
        output_type=Output.DATAFRAME,
    )

    # # Extracción de config temporal para proceso de optimización manual del ocr
    # with open('out_data/tess_config.txt', 'w') as f:
    #     f.write(tess_config)

    df_data = symbols_string_filter(phantom_ocr_filter(vertical_filter(blank_filter(df_data))))

    if df_data.shape[0] == 0:
        print('Error: No se logró identificar ninguna palabra.')
    return df_data


def get_line_group_token_boxes(df_data) -> list[dict]:
    df_data['x2'] = df_data['left'] + df_data['width']
    df_data['y2'] = df_data['top'] + df_data['height']
    df_data['id_line_group'] = df_data.apply(lambda row:
                                                'id_' + 
                                                str(row['block_num']).zfill(3) + '_' +
                                                str(row['par_num']).zfill(2) + '_' +
                                                str(row['line_num']).zfill(5), axis=1
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
        group_boundaries['id_line_group'] = g
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
            "text": groups_boundaries[x]['text'],
            "id_line_group": groups_boundaries[x]['id_line_group']
            
        },
            groups_boundaries
    )

    return list(token_line_groups_boxes)

def image_preprocess(img):
    img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    img = cv2.dilate(img, (2,1), iterations=1)
    return img

def apply_tesseract(
    data_item,
    tesseract_langs: str = "spa",
    tesseract_config: str = "with_whitelist",
    output_path: str = "",
):
    data_item = deepcopy(data_item)
    img = image_preprocess(data_item["img_bitmap"])
    image = cv2pil(img)
    print(f"Aplicando OCR sobre archivo {data_item['file_path']} ...")
    df_data =  tesseract_word_boxes(image, tesseract_langs, tesseract_config)
    
    # Checks OCR null result, if null returns -1 for jumping this image's data_item creation
    if df_data.shape[0] == 0:
        return -1

    token_boxes = get_line_group_token_boxes(df_data)
    photo_boxes = get_photo_polygons(image=img, df_data=df_data, buffer_ratio=BUFFER_RATIO)
    token_boxes = add_photo_token_boxes(token_boxes, photo_boxes)

    data_item["token_boxes"] = token_boxes

    return data_item