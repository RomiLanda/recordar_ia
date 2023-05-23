import json
import warnings
import numpy as np
import pandas as pd
from copy import deepcopy
from more_itertools import flatten
from pytorch_lightning import Trainer
from torch_geometric.loader import DataLoader
from sklearn.metrics import classification_report
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.warnings import PossibleUserWarning
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from .nn_model import Model
from .utils import save_json, load_json
from .training_utils import split_dataset, set_label_map, get_pg_graphs, MONITOR_MAP


warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=PossibleUserWarning)

SAVE_MODEL_PATH = "src/models" 
MODEL_FILE_NAME = "model"

# model parameters
HIDDEN_CHANNELS = 512
BATCH_SIZE = 32
MAX_EPOCHS = 5000

 
def train_model(data_block, save_path: str):
    train, val, test = split_dataset(data_block)
    
    label_map, inv_label_map = set_label_map(data_block)

    pg_graph_train = get_pg_graphs(train, label_map, train_flow = True)
    pg_graph_val = get_pg_graphs(val, label_map, train_flow = True)

    n_classes = len(label_map)
    n_features = pg_graph_train[0].x.shape[1]

    train_loader = DataLoader(
        pg_graph_train, batch_size=BATCH_SIZE, shuffle=False, num_workers = 16
    )

    val_loader = DataLoader(
        pg_graph_val, batch_size=BATCH_SIZE, shuffle=False, num_workers = 16
    )

    model = Model(
        train_loader, 
        val_loader,
        hidden_channels= HIDDEN_CHANNELS,
        n_features= n_features,
        n_classes= n_classes,
    )

    train_monitor = "loss"
    es_patience = 100

    monitor = MONITOR_MAP[train_monitor]["monitor"]
    mode = MONITOR_MAP[train_monitor]["mode"]

    early_stop_callback = EarlyStopping(
        monitor=monitor,
        mode=mode,
        min_delta=0.00,
        patience= es_patience,
        verbose=False,
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath= save_path,
        filename= MODEL_FILE_NAME,
        monitor=monitor,
        mode=mode,
        save_top_k=1,
    )

    trainer = Trainer(
        max_epochs= MAX_EPOCHS,
        callbacks=[
            early_stop_callback,
            checkpoint_callback
        ],
        default_root_dir = save_path,
    )
    
    trainer.fit(model)

    model_metadata = {
        "n_features": n_features,
        "n_classes": n_classes,
    }

    save_json(model_metadata, f"{save_path}/metadata.json")
    save_json(label_map, f"{save_path}/label_map.json")
    save_json(inv_label_map, f"{save_path}/inv_label_map.json")

    return train, val, test


def load_model(path_model: str, model_file_name: str):
    model_metadata = load_json(f"{path_model}/metadata.json")
    label_map = load_json(f"{path_model}/label_map.json")
    inv_label_map = load_json(f"{path_model}/inv_label_map.json")

    model_in_file = f"{path_model}/{model_file_name}.ckpt"
    
    model = Model.load_from_checkpoint(
        checkpoint_path= model_in_file,
        train_loader=None,
        val_loader=None,
        hidden_channels= HIDDEN_CHANNELS,
        n_features=model_metadata["n_features"],
        n_classes=model_metadata["n_classes"],
        class_weights=None,
    )

    trainer = Trainer(
        default_root_dir = SAVE_MODEL_PATH,
    )

    return model, trainer, label_map, inv_label_map


def data_item_predict(data_item, pred_map: dict):
    data_item = deepcopy(data_item)
    data_item["token_boxes"] = [
        token_box | pred_map[token_box["id_line_group"]]
        for token_box in data_item["token_boxes"]
    ]
    return data_item


def predict(data_block, path_model: str, model_file_name: str):
    
    model, trainer, label_map, inv_label_map = load_model(SAVE_MODEL_PATH, MODEL_FILE_NAME)

    pg_graphs = get_pg_graphs(data_block, label_map, train_flow = False)

    loader = DataLoader(
        pg_graphs, batch_size=5, shuffle=False
    )

    pred_tuples = trainer.predict(model, loader)
    preds = [pred[0].cpu().numpy() for pred in pred_tuples]
    confidences = [pred[1][0].cpu().numpy() for pred in pred_tuples]

    preds = np.hstack(preds)
    confidences = np.hstack(confidences)

    pred_labels = (inv_label_map[str(label_idx)] for label_idx in preds)
    node_ids = (
        (token_box["id_line_group"] for token_box in data["token_boxes"])
        for data in data_block
    )

    node_ids = flatten(node_ids)
    pred_map = {
        idx: {"pred_label": pred_label, "cls_conf": conf}
        for idx, pred_label, conf in zip(
            node_ids, pred_labels, confidences
        )
    }

    data_block = [
        data_item_predict(data_item, pred_map)
        for data_item in data_block
    ]

    return data_block


def show_predictions(data_block):
    y_true = []
    y_pred = []
    for data_item in data_block:
        for token in data_item['token_boxes']:
            y_true.append(token['label'])
            y_pred.append(token['pred_label'])

    print(classification_report(y_true, y_pred))


def write_output_json(data_block):
    OUTPUT_PATH = 'out_data'

    noticia_procesada = {"Diario": [],
                        "Fecha": [],
                        "Volanta": [],
                        "Título": [],
                        "Cuerpo": [],
                        "Copete": [],
                        "Destacado": [],
                        "Epígrafe": [],
                        "Firma": []}
    
    partes_noticia = noticia_procesada.keys()

    df_data_items = pd.DataFrame(data_block)

    for i, row in df_data_items.iterrows():
        json_out_file = row['file_path'].split('/')[-1].replace('.tif', '.json')
        df_token_boxes = (pd.DataFrame(row['token_boxes']))
        gp_tokens = df_token_boxes.groupby(by='pred_label')
        for label in partes_noticia:
            if label in gp_tokens.groups.keys():
                noticia_procesada[label] = ' '.join(gp_tokens.get_group(label).sort_values(by='id_line_group')['text'])
        
        with open(f'{OUTPUT_PATH}/{json_out_file}', 'w') as f:
            json.dump(noticia_procesada, f, ensure_ascii=False, indent=2)

        print(f'Archivo {json_out_file} generado correctamente!')


def process(data_block, train_flow: bool):
    if train_flow:
        _, _, test = train_model(data_block, SAVE_MODEL_PATH)
        predict_data_block = predict(test, SAVE_MODEL_PATH, MODEL_FILE_NAME)
        show_predictions(predict_data_block)

    else:
        predict_data_block = predict(data_block, SAVE_MODEL_PATH, MODEL_FILE_NAME)
        write_output_json(predict_data_block)