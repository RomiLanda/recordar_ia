import sys
import argparse
from src.src import procesar_imgs

if __name__ == '__main__':
    desc = 'Software para transcripción de notas periodísticas'

    parser = argparse.ArgumentParser(description=desc, epilog="Fundación Sadosky - Procuración del Tesoro de la Nación")
    parser.add_argument('-e', '--entrada',
                        default="input_data/",
                        help="Directorio donde se encuentran los archivos de imagen para digitalizar.",
                        type=str)
    parser.add_argument('-s', '--salida',
                        default='out_data/',
                        help="Directorio donde se guardan los resultados.",
                        type=str)

    args = parser.parse_args()
    if not args:
        print("Necesitas insertar argumentos")
    else:
        procesar_imgs(path_in=args.entrada,
            path_out=args.salida)