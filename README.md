# RecordarIA
Software para la digitalización, transcripción y clasificación de las distintas partes de notas periodisticas.

<img src="https://desafio-ia-por-la-identidad.fundacionsadosky.org.ar/wp-content/uploads/2023/02/IMG_0317-1024x683.jpg"/>

## Running 
### Ambiente y requerimientos
Deberías usar Python 3.10.6, chequea eso antes de crear tu `virtualenv`

1. Crear virtualenv usando python3 (follow https://virtualenvwrapper.readthedocs.io/en/latest/install.html)

        virtualenv <name_env>

2. Activar el virtualenv

        source <name_env>/bin/activate

3. Instalar python requirements

        pip install -r requirements.txt

4. Instalar tesseract

        sudo sh install.sh
### Data 
Para realizar el entrenamiento es necesario contar con notas previamente etiquetadas. 
Para realizar el etiquetado puede usar [Label Studio](https://labelstud.io/) y posteriormente realizar la conversión al formato indicado usando [labels-to-json](https://github.com/damianmdp/labels-to-json-diaxi).
Cada imagen de nota en formato .tif debe estar acompañada del respectivo archivo .json con las anotaciones. 
En la carpeta input data puede encontar un modelo de ejemplo.

### Run
Hay dos formas de trabajo configurables mediante la variable `TRAIN_FLOW` que se encuentra en el archivo `src.py` (su valor por default es `False`): <br />
- Para realizar el entrenamiento del modelo y guardar dicho modelo para poder ser utilizado (reemplaza al archivo del modelo existente):

        TRAIN_FLOW = True 

- Para realizar predicciones a partir de un modelo previamente entrenado (flujo por default): 

        TRAIN_FLOW = False

Una vez defina la forma de trabajo, se ejecuta desde la terminal

        python run.py
