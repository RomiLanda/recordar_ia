# recordar_ia

## Running 

Deberías usar Python 3.10.6, chequea eso antes de crear tu `virtualenv`

1. Crear virtualenv usando python3 (follow https://virtualenvwrapper.readthedocs.io/en/latest/install.html)

        virtualenv <name_env>

2. Activar el virtualenv

        source <name_env>/bin/activate

3. Instalar python requirements

        pip install -r requirements.txt

4. Instalar tesseract

        sudo sh install.sh

5. Hay dos formas de trabajo configurables mediante la variable `TRAIN_FLOW` que se encuentra en el archivo `src.py` (su valor por default es `False`): <br />
- Para realizar el entrenamiento del modelo y guardar dicho modelo para poder ser utilizado (reemplaza al archivo del modelo existente):

        TRAIN_FLOW = True 

- Para realizar predicciones a partir de un modelo previamente entrenado (flujo por default): 

        TRAIN_FLOW = False

Una vez defina la forma de trabajo, se ejecuta desde la terminal

        python run.py
