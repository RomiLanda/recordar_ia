import os
import cv2
import json
import pandas as pd
from shapely import box
from copy import deepcopy

from .debug import doc_debug
from .ocr_boxes import apply_tesseract
from .utils import get_label_tokens
from .create_graph import create_doc_graphs
from .add_features import add_features

# filtramos (usamos) las imagenes que tengan una cantidad de notas menor o igual a NEWS_QTY_TO_FILTER
# por lo que si NEWS_QTY_TO_FILTER=3, vamos a quedarnos con las imagenes que tengan hasta 3 notas

# con valor=1 usamos uni-nota - una imagen con una nota
# con valor=99 usamos multi-nota (incluidas las uni-nota) - una imagen con mas de una nota
NEWS_QTY_TO_FILTER = 1

def load_image(img_path : str):
    """
    Create DataItem
    """
    if img_path.endswith(".tif"):            
        imagen_cv = cv2.imread(img_path)
        data_item = {}
        data_item['file_path'] = img_path
        data_item['img_bitmap'] = imagen_cv
        data_item["image_shape"] =  {
            "image_height" : imagen_cv.shape[0], 
            "image_width" : imagen_cv.shape[1]
            }
        return data_item


def image_unload(data_item):
    """Release image memory
    Returns:
        DataItem: data without loaded image
    """
    data_item = deepcopy(data_item)
    data_item.pop("img_bitmap", None)

    return data_item


class Caja(object):
    """
    Defines the data structure of the labeled segment (bounding-box) as a unit.
    """
    def __init__(self, archivo, bounding_box, etiqueta, contenido):
        self.file = archivo
        self.x_1 = bounding_box['x']
        self.y_1 = bounding_box['y']
        self.x_2 = bounding_box['x'] + bounding_box['width'] 
        self.y_2 = bounding_box['y'] + bounding_box['height']
        self.box = (bounding_box['x'], bounding_box['y'], bounding_box['x'] + bounding_box['width'], bounding_box['y'] + bounding_box['height'])
        self.polygon = box(bounding_box['x'], bounding_box['y'], bounding_box['x'] + bounding_box['width'], bounding_box['y'] + bounding_box['height'])
        self.content = contenido
        self.label = etiqueta
        
    def to_json(self):
        return self.__dict__
    
    def to_pd_series(self):
        return pd.Series(self.__dict__)


def get_annotations(json_path):
    if json_path.endswith(".json"):
        try:
            with open(json_path, "r") as json_file:
                return json.load(json_file)
        except FileNotFoundError:
            print(f'Archivo: {json_path} | Archivo no encontrado  | Estado ERROR!')

    return {}


def get_segments_from_annotations(data_item, annotations, json_path):
    """
    Extracts each segment of each article in the json file
    """    
    data_item['segments'] = []
    if annotations is not None:
        print(f"Diario: {annotations['Diario']}")

        file_path = json_path.replace('.json', '.tif')
        for segmento in ['Diario', 'Fecha']:
            try:
                caja = Caja(file_path, annotations[segmento]['bounding_box'], segmento, annotations[segmento]['texto'])
                data_item['segments'].append(caja.to_json())
                #print(f'Archivo: {caja.file} | Segmento: {segmento}  | Estado OK!')
            except (ValueError, KeyError) as e:
                print(f'Archivo: {json_path} | Segmento: {segmento}  | Estado ERROR no key/value:{str(e)}!')

        for nota in annotations['Notas']:
            for segmento in nota:
                if not isinstance(nota[segmento], bool) and nota[segmento]:
                    for detalle in nota[segmento]:
                        try:
                            caja = Caja(file_path, detalle['bounding_box'], segmento, detalle['texto'])
                            data_item['segments'].append(caja.to_json())
                            #print(f'Archivo: {caja.file} | Segmento: {segmento}  | Estado OK!')
                        except (ValueError, KeyError) as e:
                            print(f'Archivo: {json_path} | Segmento: {segmento}  | Estado ERROR no key/value: {str(e)}!')
    
    return data_item


def create_data_block(INPUT_DATA, OUTPUT_DATA, debug = False):
    data_block = []
    print(f"Cantidad de imágenes a procesar: {len(os.listdir(INPUT_DATA))/2}")
    for filename in os.listdir(INPUT_DATA):
        file_path = f"{INPUT_DATA}{filename}"
        json_path = file_path.replace('.tif','.json')
        annotations = get_annotations(json_path)
        if not NEWS_QTY_TO_FILTER or (annotations and len(annotations['Notas']) <= NEWS_QTY_TO_FILTER):
            data_item = load_image(file_path)
            if data_item:
                data_item = get_segments_from_annotations(data_item, annotations, json_path)
                data_item = apply_tesseract(data_item, output_path=OUTPUT_DATA)
                data_item = add_features(data_item)
                data_item = get_label_tokens(data_item)
                data_item = create_doc_graphs(data_item)
                if debug:
                    doc_debug(data_item, OUTPUT_DATA)
                data_item = image_unload(data_item)
                data_block.append(data_item)
    print(f"Se procesaron {len(data_block)} notas")
    return data_block