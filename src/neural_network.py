import os
from sklearn.metrics import classification_report
import numpy as np
from .training_utils import split_dataset, set_label_map, get_pg_graphs, MONITOR_MAP
from .nn_model import Model
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning import Trainer
from torch_geometric.loader import DataLoader
from copy import deepcopy
from more_itertools import flatten
from pytorch_lightning.utilities.warnings import PossibleUserWarning
import torch

import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=PossibleUserWarning)

SAVE_MODEL_PATH = "./models/model.pt"

# model parameters
HIDDEN_CHANNELS = 512
BATCH_SIZE = 32
MAX_EPOCHS = 5000


def create_model(pg_graph_train, pg_graph_val, pg_graph_test, n_classes):
    n_features = pg_graph_train[0].x.shape[1]

    train_loader = DataLoader(
        pg_graph_train, batch_size=BATCH_SIZE, shuffle=False, num_workers = 16
    )

    val_loader = DataLoader(
        pg_graph_val, batch_size=BATCH_SIZE, shuffle=False, num_workers = 16
    )

    test_loader = DataLoader(
        pg_graph_test, batch_size=BATCH_SIZE, shuffle=False, num_workers = 16
    )

    model = Model(
        train_loader, 
        val_loader,
        hidden_channels= HIDDEN_CHANNELS,
        n_features= n_features,
        n_classes= n_classes,
    )

    return model


def train_model(label_map, train, val, test, use_existing_model):
    n_classes = len(label_map)

    pg_graph_train = get_pg_graphs(train, label_map)
    pg_graph_val = get_pg_graphs(val, label_map)
    pg_graph_test = get_pg_graphs(test, label_map)

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

    model = create_model(pg_graph_train, pg_graph_val, pg_graph_test, n_classes)

    if use_existing_model and os.path.isfile(SAVE_MODEL_PATH):
        print(f"Loading existing model {SAVE_MODEL_PATH}")
        model_state_dict = torch.load(SAVE_MODEL_PATH)
        model.load_state_dict(model_state_dict)
    else:
        print("Creating new model")

    trainer = Trainer(
        max_epochs= MAX_EPOCHS,
        callbacks=[
            early_stop_callback,
        ]
    )

    if not use_existing_model:
        print("Training model")
        trainer.fit(model)

    if not os.path.isfile(SAVE_MODEL_PATH):
        # guardamos el último modelo entrenado
        print("Saving model")
        torch.save(model.state_dict(), SAVE_MODEL_PATH)
    
    return model, trainer


def data_item_predict(data_item, pred_map: dict):
    data_item = deepcopy(data_item)
    data_item["token_boxes"] = [
        token_box | pred_map[token_box["id_line_group"]]
        for token_box in data_item["token_boxes"]
    ]
    return data_item


def predict(trainer, model, data_block, label_map, inv_label_map):
    pg_graphs = get_pg_graphs(data_block, label_map)

    loader = DataLoader(
        pg_graphs, batch_size=5, shuffle=False
    )

    pred_tuples = trainer.predict(model, loader)
    preds = [pred[0].cpu().numpy() for pred in pred_tuples]
    confidences = [pred[1][0].cpu().numpy() for pred in pred_tuples]

    preds = np.hstack(preds)
    confidences = np.hstack(confidences)

    pred_labels = (inv_label_map[label_idx] for label_idx in preds)
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


def show_predictions(predict_data_block):
    y_true = []
    y_pred = []
    for data_item in predict_data_block:
        for token in data_item['token_boxes']:
            y_true.append(token['label'])
            y_pred.append(token['pred_label'])

    print(classification_report(y_true, y_pred))


def unify_text_by_label(predict_data_block):
    import pdb; pdb.set_trace()
    notes = []
    # usar id_line_group, text, pred_label
    for data_block in predict_data_block:
        new_note = {}
        token_boxes = data_block['token_boxes']
        last_label = None
        content = ""
        for token_box in token_boxes:
            token_box['id_line_group'].split("_")
            # if token_box['pred_label'] == last_label:
            #     content = content + token_box['text']
            # else:
            print(token_box['pred_label'])
            print(token_box['text'])
        notes.append(new_note)


def process(data_block, use_existing_model=True):
    train, val, test = split_dataset(data_block)
    label_map, inv_label_map = set_label_map(data_block)

    model, trainer = train_model(label_map, train, val, test, use_existing_model)
    predict_data_block = predict(trainer, model, test, label_map, inv_label_map)
    show_predictions(predict_data_block)
    unify_text_by_label(predict_data_block)