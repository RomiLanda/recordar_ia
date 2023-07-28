import pandas as pd

from .neural_network import process
from .load_data import create_data_block

# SET WORKFLOW
TRAIN_FLOW = True
DEBUG = True

def procesar_imgs(path_in, path_out, train_flow = TRAIN_FLOW): #TODO agregar corrector jurídico si es necesario
    data = create_data_block(path_in, path_out, train_flow, debug = DEBUG)
    
    if DEBUG:
        report_file = 'data_block_train.csv' if train_flow == True else 'data_block_pred.csv'
        pd.DataFrame(data=data).to_csv(path_out + report_file, index=False)

        print(f'Se creo el archivo de reporte --> {report_file}')

    process(data, train_flow)
    print(f'Procesamiento finalizado. Los resultados fueron guardados en el directorio: {path_out}.')