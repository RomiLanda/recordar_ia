from copy import deepcopy
from itertools import chain
from shapely import box, geometry
from shapely.ops import unary_union
import cv2
import numpy as np

def clear_abnormal_ratio_photos(boxes_list):
    """
    Returns polygon list filtered by aspect ratio.
    """
    MAX_PHOTO_ASPECT_RATIO = 3
    clean_boxes_list = []
    for p in boxes_list:
        x1, y1, x2, y2 = map(int, p.bounds)
        w = x2 - x1
        h = y2 - y1
        if ((max(h, w) / min(h, w)) < MAX_PHOTO_ASPECT_RATIO):
            clean_boxes_list.append(p)
    return clean_boxes_list

def clear_abnormal_data_photos(boxes_list, image):
    """
    Returns polygon list filtered by data statistics.
    """
    MAX_MEAN_PIXELS_DATA = 200
    clean_boxes_list = []

    for p in boxes_list:
        x1, y1, x2, y2 = map(int, p.bounds)
        crop_image = image[y1:y2, x1:x2]
        mean_data = np.mean(crop_image)
        if (mean_data < MAX_MEAN_PIXELS_DATA):
            clean_boxes_list.append(p)

    return clean_boxes_list


def get_photo_polygons(image, df_data, buffer_ratio=1.8):
    """
    Returns list of polygons with high probabilities of being a photograph, from image and pytesseract dataframe.
    """
    
    MIN_PHOTO_AREA_PERC = 3
    MAX_PHOTO_AREA_PERC = 90
    DIL_ITER=2
    K_SIZE_BLUR=(7,7)
    K_SIZE_DIL=(4,4)
    FILL_COLOR = (255, 255, 255)

    polygons = []
    image = deepcopy(cv2.bitwise_not(image))
    image_height, image_width = image.shape
    image_area = image_height * image_width
    
    for token in df_data.iterrows():
        w, h = int(token[1].width * buffer_ratio), int(token[1].height * buffer_ratio)
        x, y = int(token[1].left - (token[1].width * (buffer_ratio-1) / 2)), int(token[1].top - (token[1].height * (buffer_ratio-1) / 2))
        cv2.rectangle(image, (x, y), (x + w, y + h), FILL_COLOR, -1)

    blur = cv2.GaussianBlur(image, K_SIZE_BLUR, 10)
    thresh = cv2.threshold(blur, 250, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    dilate = cv2.erode(thresh, cv2.getStructuringElement(cv2.MORPH_RECT, K_SIZE_DIL), iterations=DIL_ITER)
    contours = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contours = contours[0] if len(contours) == 2 else contours[1]

    for c in contours:
        x_min = c[c[:, :, 0].argmin()][0][0]
        y_min = c[c[:, :, 1].argmin()][0][1]
        x_max = c[c[:, :, 0].argmax()][0][0]
        y_max = c[c[:, :, 1].argmax()][0][1]
        polygons.append(box(x_min, y_min, x_max, y_max))
    
    polygons = [p for p in polygons if ((round((p.area/image_area)*100, 2) > MIN_PHOTO_AREA_PERC) & (round((p.area/image_area)*100, 2) < (MAX_PHOTO_AREA_PERC)))]
    polygons_union = unary_union(polygons)
    
    if isinstance(polygons_union, geometry.multipolygon.MultiPolygon): 
        polygons = list(polygons_union.geoms)
        polygons = [p for p in polygons if ((round((p.area/image_area)*100, 2) > MIN_PHOTO_AREA_PERC) & (round((p.area/image_area)*100, 2) < (MAX_PHOTO_AREA_PERC)))]
    elif isinstance(polygons_union, geometry.polygon.Polygon):
        polygons = []
        if ((round((polygons_union.area/image_area)*100, 2) > MIN_PHOTO_AREA_PERC) & (round((polygons_union.area/image_area)*100, 2) < (MAX_PHOTO_AREA_PERC))):
            polygons.append(polygons_union)        
    
    polygons = [box(*p.bounds) for p in polygons]
    photo_poly_candidates = clear_abnormal_data_photos(clear_abnormal_ratio_photos(polygons), image)

    return photo_poly_candidates


def add_photo_token_boxes(token_boxes, photo_boxes):
    """
    Returns list of polygons without token_boxes contained in photo_boxes.
    """
    if len(photo_boxes) > 0:
        tokens_to_delete = []

        for ph_box in photo_boxes:
            for token in token_boxes:
                if ph_box.contains(token['box_polygon']):
                    tokens_to_delete.append(token)
        
        for token in tokens_to_delete:
            token_boxes.remove(token)
        
        token_boxes = map(lambda x: x, token_boxes)

        token_photo_boxes = map(
            lambda x: {
                "top": int(x[1].bounds[1]),
                "left": int(x[1].bounds[0]),
                "box": list(map(int, x[1].bounds)),
                "box_polygon": x[1],
                "box_area": int(x[1].area),
                "box_height": int(x[1].bounds[3] - x[1].bounds[1]),
                "box_width": int(x[1].bounds[2] - x[1].bounds[0]),
                "x_position": int(x[1].bounds[0]),
                "y_position": int(x[1].bounds[1]),
                "text": 'photo_box',
                "id_line_group": 'id_photo_' + str(x[0])
            },
                enumerate(photo_boxes)
        )
        
        token_boxes = map(lambda x: x, chain(token_boxes, token_photo_boxes))
        
    return list(token_boxes)