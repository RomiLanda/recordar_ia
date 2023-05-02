import statistics
import dateparser
import re


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


def get_attributes_text(data_item):
    for box in data_item['token_boxes']:
        text = box['text']
        box['caps_words_ratio'] = caps_words_ratio(text)
        box['is_date'] = is_date(text)
        box['number_presence'] = number_presence(text)
    return data_item


def normalize_positions(data_item):
    image_height = data_item['image_shape']['image_height']
    image_width = data_item['image_shape']['image_width']
    for box in data_item['token_boxes']:
        box['x_position_normalized'] = (box['x_position'] / image_width) * 100
        box['y_position_normalized'] = (box['y_position'] / image_height) * 100
        box['box_area_normalized'] = (box['box_area'] / (image_height * image_width)) * 100
    return data_item


def add_features(data_item):
    data_item = mode_normalized_height(data_item)
    data_item = normalize_positions(data_item)
    data_item = get_attributes_text(data_item)
    return data_item
