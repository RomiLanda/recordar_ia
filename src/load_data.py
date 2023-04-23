import os
import cv2
import json
import pandas as pd
from shapely import LineString, Point, Polygon, box

from .debug import doc_debug
from .ocr_boxes import apply_tesseract
from .utils import get_label_tokens, image_unload
from .create_graph import create_doc_graphs


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


class Caja(object):
    """
    Defines the data structure of the labeled segment (bounding-box) as a unit.
    """
    def __init__(self, archivo, bounding_box, etiqueta, contenido):
        self.file = archivo
        self.x_1 = bounding_box['x']
        self.y_1 = bounding_box['y']
        self.x_2 = bounding_box['x'] + bounding_box['w'] 
        self.y_2 = bounding_box['y'] + bounding_box['h']
        self.box = (bounding_box['x'], bounding_box['y'], bounding_box['x'] + bounding_box['w'], bounding_box['y'] + bounding_box['h'])
        self.polygon = box(bounding_box['x'], bounding_box['y'], bounding_box['x'] + bounding_box['w'], bounding_box['y'] + bounding_box['h'])
        self.content = contenido
        self.label = etiqueta
        
    def to_json(self):
        return self.__dict__
    
    def to_pd_series(self):
        return pd.Series(self.__dict__)


def get_segments_from_annotations(data_item):
    """
    Extracts each segment of each article in the json file
    """    
    json_path = data_item['file_path'].replace('.tif','.json')
    data_item['segments'] = []

    if json_path.endswith(".json"):
        with open(json_path, "r") as json_file:
            datos = json.load(json_file)

        print(datos['Diario'])

        try:
            caja = Caja(json_path.replace('.json', '.tif'), datos['Diario']['bounding_box'], 'Diario', datos['Diario']['texto'])
            data_item['segments'].append(caja.to_json())
            #print(f'Archivo: {caja.file} | Segmento: {segmento}  | Estado OK!')
        except:
            print(f'Archivo: {json_path} | Segmento: Diario  | Estado ERROR!')

        try:
            caja = Caja(json_path.replace('.json', '.tif'), datos['Fecha']['bounding_box'], 'Fecha', datos['Fecha']['texto'])
            data_item['segments'].append(caja.to_json())
            #print(f'Archivo: {caja.file} | Segmento: {segmento}  | Estado OK!')
        except:
            print(f'Archivo: {json_path} | Segmento: Fecha  | Estado ERROR!')

        for nota in datos['Notas']:
            for segmento in nota:
                for detalle in nota[segmento]:
                    try:
                        caja = Caja(json_path.replace('.json', '.tif'), detalle['bounding_box'], segmento, detalle['texto'])
                        data_item['segments'].append(caja.to_json())
                        #print(f'Archivo: {caja.file} | Segmento: {segmento}  | Estado OK!')
                    except:
                        print(f'Archivo: {json_path} | Segmento: {segmento}  | Estado ERROR!')
    
    return data_item


def create_data_block(INPUT_DATA, OUTPUT_DATA, debug = False):
    data_block = []
    for filename in os.listdir(INPUT_DATA):
        if filename.endswith(".tif"):
            file_path = f"{INPUT_DATA}{filename}"
            data_item = load_image(file_path)
            data_item = get_segments_from_annotations(data_item)
            data_item = apply_tesseract(data_item, output_path=OUTPUT_DATA)
            data_item = get_label_tokens(data_item)
            data_item = create_doc_graphs(data_item)
            if debug:
                doc_debug(data_item, OUTPUT_DATA)
            data_item = image_unload(data_item)
            data_block.append(data_item)
        
    return data_block