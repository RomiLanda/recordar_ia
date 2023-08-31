import cv2
import numpy as np
from shapely import box, geometry, unary_union
from copy import deepcopy
from itertools import groupby
from pytesseract import image_to_data, Output
from .utils import cv2pil, blank_filter, vertical_filter, phantom_ocr_filter, symbols_string_filter
from .get_photos import add_photo_token_boxes, get_photo_polygons
import pickle

tess_configs = {
    "default": "--psm 11",
    "psm3": "--psm 3",
    "psm4": "--psm 4",
    "psm5": "--psm 5",
    "psm6": "--psm 6",
    "psm12": "--psm 12",
    "with_whitelist": r'-c tessedit_char_whitelist="AÁBCDEÉFGHIÍJKLMNÑOÓPQRSTUÚVWXYZaábcdeéfghiíjklmnñoópqrstuúvwxyz0123456789 -_/.,:;()" --psm 11',
}

BUFFER_RATIO = 1.8

def dump_for_debug(variable, filename):
    with open(f'out_data/{filename}.pkl', 'wb') as f:
        pickle.dump(variable, f)


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


def get_token_boxes(df_data) -> list[dict]:
    """
    Returns token_boxes list from PyTesseract dataframe.
    """
    
    df_data['x2'] = df_data['left'] + df_data['width']
    df_data['y2'] = df_data['top'] + df_data['height']
    df_data = df_data.rename(columns={'left': 'x1', 'top': 'y1'})

    token_boxes = df_data.apply(
        lambda row: {
            "top": row['y1'],
            "left": row['x1'],
            "box": (row['x1'],
                    row['y1'],
                    row['x2'],
                    row['y2']),
            "box_polygon": box(row['x1'],
                               row['y1'],
                               row['x2'],
                               row['y2']),
            "box_area": row['x1'] * row['y1'],
            "box_height": row['y2'] - row['y1'],
            "box_width" : row['x2'] - row['x1'],
            "x_position": row['x1'],
            "y_position": row['y1'],
            "text": row['text'],
        }, 
            axis=1
    ).to_list()
    return token_boxes

def get_multipolygon(token_boxes) -> geometry.MultiPolygon:
    """
    Returns shapely.multipolygon object from tokens.
    """
    polygons = map(lambda x: x['box_polygon'], token_boxes)
    return geometry.MultiPolygon(polygons)

def buffer_polygons(multipoly, buffer_w, buffer_h):
    """
    Applies geometric buffer over polygons. Extensions are
    polygon height proportional with W and H separated ratios.
    """
    multipoly_scaled = []
    for p in multipoly.geoms:
        x1, y1, x2, y2 = p.bounds
        ref_h = y2 - y1
        multipoly_scaled.append(box(x1 - round(ref_h * buffer_w, 0),
                                    y1 - round(ref_h * buffer_h, 0),
                                    x2 + round(ref_h * buffer_w, 0),
                                    y2 + round(ref_h * buffer_h, 0)))
    return geometry.MultiPolygon(multipoly_scaled)

def get_h_avg_rectangle(poly) -> geometry.box:
    """
    Returns rectangle simplified from complex polygon 
    with average height.
    """
    x_coords = [v[0] for v in poly.exterior.coords]
    y_coords = [v[1] for v in poly.exterior.coords]
    new_x1 = np.min(x_coords)
    new_x2 = np.max(x_coords)
    third_h_mean = round((np.max(y_coords) - np.min(y_coords))/3, 0)
    centroid_y = round((np.max(y_coords) + np.min(y_coords))/2, 0) #round(np.mean(y_coords), 0)
    new_y1 = centroid_y - third_h_mean
    new_y2 = centroid_y + third_h_mean
    new_y1, new_y2
    return box(new_x1, new_y1, new_x2, new_y2)

def simplify_nested_polygons(multipoly) -> geometry.MultiPolygon:
    """
    Returns rectangle simplified from complex polygon.
    """ 
    multipoly_simplified = []
    if isinstance(multipoly, geometry.multipolygon.MultiPolygon):
        for p in multipoly.geoms:
            x1, y1, x2, y2 = p.bounds
            multipoly_simplified.append(box(x1, y1, x2, y2))
    elif isinstance(multipoly, geometry.polygon.Polygon):
        x1, y1, x2, y2 = multipoly.bounds
        multipoly_simplified.append(box(x1, y1, x2, y2))
    return geometry.MultiPolygon(multipoly_simplified)

def filter_tokens(token_boxes, field, perc_range=0.05) -> list[dict]:
    """
    Returns token_boxes filtered by 'perc_range' range of
    values over 50-percentile of 'field' parameter.
    """
    values = list(map(lambda x: x[field], token_boxes))
    center_value = np.percentile(values, 50)
    low_limit = center_value * (1 + perc_range)
    high_limit = center_value * (1 - perc_range)
    return list(filter(lambda x: low_limit < x[field] < high_limit, token_boxes))

def filter_multipolygons(multipoly, perc_range_w=0.2, perc_range_h=1) -> geometry.MultiPolygon:
    """
    Returns token_boxes filtered by 'perc_range_w' and 'perc_range_h'
    range of values over 50-percentile of Width and Height parameters.
    """
    center_value_h = np.percentile(list(map(lambda x: x.bounds[1]-x.bounds[3], multipoly.geoms)), 50)
    center_value_w = np.percentile(list(map(lambda x: x.bounds[0]-x.bounds[2], multipoly.geoms)), 50)
    low_limit_h = center_value_h * (1 + perc_range_h)
    high_limit_h = center_value_h * (1 - perc_range_h)    
    low_limit_w = center_value_w * (1 + perc_range_w)
    high_limit_w = center_value_w * (1 - perc_range_w)    
    return geometry.MultiPolygon(list(filter(lambda x: (low_limit_h < x.bounds[1]-x.bounds[3] < high_limit_h) & (low_limit_w < x.bounds[0]-x.bounds[2] < high_limit_w), multipoly.geoms)))

def get_line_id_token_boxes(token_boxes) -> list[dict]:
    """
    Creates 'id_line_group' field from buffered polygons intersects
    for Line aggregation.
    """
    mutipolygon = get_multipolygon(token_boxes)
    multipolygon_buffered_for_lines = unary_union(buffer_polygons(mutipolygon, 0.7, -0.1))
    multipolygon_lines = geometry.MultiPolygon(sorted([get_h_avg_rectangle(p) for p in multipolygon_buffered_for_lines.geoms], key=lambda poly: poly.centroid.y))

    #dump_for_debug(mutipolygon, 'multipolygon')
    #dump_for_debug(multipolygon_buffered_for_lines, 'multipolygon_buffered_for_lines')
    #dump_for_debug(multipolygon_lines, 'multipolygon_lines')
    
    token_boxes = [{**x, 'id_line_group': 'Indefinido'} for x in token_boxes]

    for i, line_box in enumerate(multipolygon_lines.geoms):
        token_boxes = list(
            map(lambda x: {
                    **x,
                    'id_line_group': f"id_{str(i).zfill(5)}"
                } if line_box.intersects(x['box_polygon']) else {**x}
                , token_boxes
                )
            )
    return token_boxes

def get_par_id_token_boxes(token_boxes) -> list[dict]:
    """
    Creates 'id_par_group' field from buffered polygons intersects
    for Paragraph aggregation.
    """
    mutipolygon = get_multipolygon(token_boxes)
    multipolygon_par = simplify_nested_polygons(unary_union(buffer_polygons(filter_multipolygons(mutipolygon), -0.8, 4)))
    dump_for_debug(multipolygon_par, 'multipolygon_par')

    token_boxes = [{**x, 'id_par_group': 'Indefinido'} for x in token_boxes]

    for i, par_box in enumerate(multipolygon_par.geoms):
        token_boxes = list(
            map(lambda x: {
                    **x,
                    'id_par_group': f"id_{str(i).zfill(3)}"
                } if par_box.intersects(x['box_polygon']) else {**x}
                , token_boxes
                )
            )
    return token_boxes

def merge_line_token_boxes(token_boxes) -> list[dict]:
    """
    Returns token_boxes with aggregation by line.
    """
    merged_token_boxes = []
    for id_line_group, group in groupby(token_boxes, key=lambda x: x['id_line_group']):
        group_poly = geometry.MultiPolygon([x['box_polygon'] for x in deepcopy(group)])
        x1, y1, x2, y2 = group_poly.bounds
        merged_token_boxes.append({
            'top': y1,
            'left': x1,
            'box': (x1, y1, x2, y2),
            'box_polygon': box(x1, y1, x2, y2),
            'box_area': group_poly.area,
            'box_height': y2 - y1,
            'box_width': x2 - x1,
            'x_position': x1,
            'y_position': y1,
            'text': " ".join(x['text'] for x in deepcopy(group)),
            'id_line_group': id_line_group,
            }
            )
    return merged_token_boxes

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

    token_boxes = get_token_boxes(df_data)
    token_boxes = get_line_id_token_boxes(token_boxes)
    token_boxes = merge_line_token_boxes(token_boxes)
    token_boxes = get_par_id_token_boxes(token_boxes)
    photo_boxes = get_photo_polygons(image=img, df_data=df_data, buffer_ratio=BUFFER_RATIO)
    token_boxes = add_photo_token_boxes(token_boxes, photo_boxes)

    data_item["token_boxes"] = token_boxes

    return data_item