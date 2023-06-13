import re
import statistics
import dateparser
import pandas as pd


def mode_height(data_item):
    total_height = []
    for box in data_item['token_boxes']:
        total_height.append(box['box_height'])
    return statistics.mode(total_height)
        

def mode_normalized_height(data_item):
    mode = mode_height(data_item)
    for box in data_item['token_boxes']:
        box['mode_normalized_height'] = box['box_height'] / mode
    return data_item


def contains_date(text: str) -> bool:
    parsed_date = dateparser.parse(text, settings={"STRICT_PARSING": False})
    if not parsed_date:
        return False
    return True


def caps_words_ratio(text: str):
    return len(re.findall(r"\b[A-Z][a-z]*", text))/len(text.split()) if len(text.split()) > 0 else 0

def get_alpha_ratio(text: str):
    if len(text) == 0:
        return 0
    alpha_counter = sum(1 for char in text if char.isalpha())
    return alpha_counter/len(text)

def number_presence(text: str):
    return len(re.findall(r"\d", text)) > 0


def is_capitalized(text: str):
    return len(re.findall(r"^[A-Z][a-z]*", text)) > 0


def get_attributes_text(data_item):
    """
    Adds quantitative features derived from text.
    """
    for box in data_item['token_boxes']:
        text = box['text']
        box['caps_words_ratio'] = caps_words_ratio(text)
        box['contains_date'] = contains_date(text)
        box['number_presence'] = number_presence(text)
        box['is_capitalized'] = is_capitalized(text)
        box['text_length'] = len(text.strip())
        box['words_qty'] = len(text.strip().split(' '))
        box['alpha_ratio'] = get_alpha_ratio(text)
    return data_item


def normalize_positions(data_item):
    """
    Adds position features normalized to image size.
    """
    image_height = data_item['image_shape']['image_height']
    image_width = data_item['image_shape']['image_width']
    for box in data_item['token_boxes']:
        box['x_position_normalized'] = (box['x_position'] / image_width)
        box['y_position_normalized'] = (box['y_position'] / image_height)
        box['box_area_normalized'] = (box['box_area'] / (image_height * image_width))
        box['x_centroid_normalized'] = box['box_polygon'].centroid.x / image_width
        box['y_centroid_normalized'] = box['box_polygon'].centroid.y / image_height
    return data_item


def create_categories(data_item, attribute: str, bins=10):
    """
    Creates categories for attribute segmentation.
    In case of photo token, it is excluded to avoid unrealistic statistics.
    """
    all_boxes_values = [box[attribute] for box in data_item['token_boxes'] if box['text'] != 'photo_box']
    return pd.cut(all_boxes_values, bins=bins, labels=range(bins))


def get_width_category(data_item):
    """
    Adds 'width_category' that groups width in N number of categories (BINS).
    In case of photo token, assings value -1.
    """
    BINS = 10
    width_categories = create_categories(data_item, attribute='box_width', bins=BINS)
    for i, box in enumerate(data_item['token_boxes']):
        if not box['text'] == 'photo_box':
            box['width_category'] = width_categories[i] / BINS
        else:
            box['width_category'] = -1
    return data_item


def get_height_category(data_item):
    """
    Adds 'height_category' that groups height in N number of categories (BINS).
    In case of photo token, assings value -1.
    """
    BINS = 7
    height_categories = create_categories(data_item, attribute='box_height', bins=BINS)
    for i, box in enumerate(data_item['token_boxes']):
        if not box['text'] == 'photo_box':
            box['height_category'] = height_categories[i] / BINS
        else:
            box['height_category'] = -1
    return data_item

def get_label_candidate(data_item):
    """
    This function creates 'label_candidate' field that suggests a token label
    based on a set of rules about existing features. Later this function could
    be replaced by a Decision Tree Classifier.
    """
    HEIGHT_BOTTOM_THRESH = 0.1
    HEIGHT_TOP_THRESH = 0.5
    LENGTH_LOW_THRESH = 3
    ALPHA_RATIO_THRESH = 0.7
    TITLE_Y_TOP_THRESH = 0.5
    MARGIN_THRESH = 0.03

    for box in data_item['token_boxes']:
        if box['text'] == 'photo_box':
            box['label_candidate'] = 'Fotografía'
        elif box['x_position_normalized'] < MARGIN_THRESH or box['x_position_normalized'] > 1 - MARGIN_THRESH or box['y_position_normalized'] < MARGIN_THRESH or box['y_position_normalized'] > 1 - MARGIN_THRESH:
            box['label_candidate'] = 'Ruido'
        elif box['contains_date'] and box['number_presence']:
            box['label_candidate'] = 'Fecha'
        elif box['text_length'] <= LENGTH_LOW_THRESH or box['alpha_ratio'] < ALPHA_RATIO_THRESH:
            box['label_candidate'] = 'Ruido'
        elif box['caps_words_ratio'] == 1 and box['words_qty'] >= 2 and not box['number_presence']:
            box['label_candidate'] = 'Firma'
        elif box['height_category'] >= HEIGHT_TOP_THRESH and box['y_position_normalized'] < TITLE_Y_TOP_THRESH:
            box['label_candidate'] = 'Título'
        elif box['height_category'] <= HEIGHT_BOTTOM_THRESH:
            box['label_candidate'] = 'Cuerpo'
        else:
            box['label_candidate'] = 'Indefinido'
        
    return data_item

def add_features(data_item):
    data_item = mode_normalized_height(data_item)
    data_item = normalize_positions(data_item)
    data_item = get_attributes_text(data_item)
    data_item = get_width_category(data_item)
    data_item = get_height_category(data_item)
    data_item = get_label_candidate(data_item)
    return data_item
