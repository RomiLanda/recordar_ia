import pandas as pd

def procesar_imgs(path_in, path_out): #TODO agregar corrector jurídico si es necesario
    data = [] # 
    noticia_procesada = {"Diario": [],
                                 "Fecha": [],
                                 "Volanta": [],
                                 "Título": [],
                                 "Cuerpo": [],
                                 "Copete": [],
                                 "Destacado": [],
                                 "Epígrafe": []}
    
    reporte = pd.DataFrame(data=data)
    reporte.to_csv(path_out + 'reporte.csv', index=False)
    print(f'Procesamiento finalizado. Los resultados fueron guardados en el directorio: {path_out}.')