import pandas as pd
from .load_data import create_data_block
from .neural_network import process

TRAIN_FLOW = False
DEBUG = True

def procesar_imgs(path_in, path_out, train_flow = TRAIN_FLOW): #TODO agregar corrector jurídico si es necesario
    data = create_data_block(path_in, path_out, train_flow, debug = DEBUG)
    processed_data = process(data, train_flow)
    
    reporte = pd.DataFrame(data=data)
    reporte.to_csv(path_out + 'reporte.csv', index=False)
    print(f'Procesamiento finalizado. Los resultados fueron guardados en el directorio: {path_out}.')