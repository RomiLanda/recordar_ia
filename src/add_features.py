import re
import statistics
import dateparser
import pandas as pd


def media_height(data_item):
    total_height = []
    for box in data_item['token_boxes']:
        total_height.append(box['box_height'])
    return statistics.mode(total_height)
        

def mode_normalized_height(data_item):
    mode = media_height(data_item)
    for box in data_item['token_boxes']:
        box['mode_normalized_height'] = box['box_height'] / mode
    return data_item


def is_date(text: str) -> bool:
    parsed_date = dateparser.parse(text, settings={"STRICT_PARSING": True})
    if not parsed_date:
        return False
    return True


def caps_words_ratio(text: str):
    return len(re.findall(r"\b[A-Z][a-z]*", text))/len(text.split()) if len(text.split()) > 0 else 0


def number_presence(text: str):
    return True if len(re.findall(r"\d", text)) > 0 else False


def is_capitalized(text: str):
    return True if len(re.findall(r"^[A-Z][a-z]*", text)) > 0 else False


def is_photo(text: str):
    return True if text == 'photo_box' else False


def get_attributes_text(data_item):
    for box in data_item['token_boxes']:
        text = box['text']
        box['caps_words_ratio'] = caps_words_ratio(text)
        box['is_date'] = is_date(text)
        box['number_presence'] = number_presence(text)
        box['is_capitalized'] = is_capitalized(text)
        box['is_photo'] = is_photo(text)
    return data_item


def normalize_positions(data_item):
    image_height = data_item['image_shape']['image_height']
    image_width = data_item['image_shape']['image_width']
    for box in data_item['token_boxes']:
        box['x_position_normalized'] = (box['x_position'] / image_width)
        box['y_position_normalized'] = (box['y_position'] / image_height)
        box['box_area_normalized'] = (box['box_area'] / (image_height * image_width))
    return data_item


def create_categories(data_item, attribute: str, bins=10):
    all_boxes_values = []
    for box in data_item['token_boxes']:
        all_boxes_values.append(box[attribute])
    return pd.cut(all_boxes_values, bins=bins, labels=range(bins))


def get_width_category(data_item):
    width_categories = create_categories(data_item, attribute='box_width', bins=10)
    for i, box in enumerate(data_item['token_boxes']):
        box['width_category'] = width_categories[i] / 10
    return data_item


def get_height_category(data_item):
    height_categories = create_categories(data_item, attribute='box_height', bins=7)
    for i, box in enumerate(data_item['token_boxes']):
        box['height_category'] = height_categories[i] / 7
    return data_item


def add_features(data_item):
    data_item = mode_normalized_height(data_item)
    data_item = normalize_positions(data_item)
    data_item = get_attributes_text(data_item)
    data_item = get_width_category(data_item)
    data_item = get_height_category(data_item)
    return data_item
