import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx
from typing import Iterator, Iterable
from more_itertools import unique_everseen
import numpy as np
from more_itertools import flatten
from pytorch_lightning.utilities.warnings import PossibleUserWarning
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=PossibleUserWarning)


NODE_FEATURES = [
    "x_position_normalized",
    "y_position_normalized",
    "box_area_normalized",
    "caps_words_ratio",
    "is_date",
    "number_presence",
    "is_capitalized",
    "width_category",
    "height_category"
]

MONITOR_MAP = {
    "f1":  {
        "monitor": "val_f1",
        "mode": "max",
    },
    "loss":  {
        "monitor": "val_loss",
        "mode": "min",
    }
}

def split_dataset(data_block, train_size: float = 0.6, val_size: float = 0.2, test_size: float = 0.2) -> tuple:
    """
    Split the data_block into train, val and test sets. The train set will be used to train the model, the val set will be used to validate the model during training and the test set will be used to test the model after training.

    Args:
        train_size (float, optional): Percentage of the data_block that will be used for training. Defaults to 0.8.
        val_size (float, optional): Percentage of the data_block that will be used for validation. Defaults to 0.1.
        test_size (float, optional): Percentage of the data_block that will be used for testing. Defaults to 0.1.

    Returns:
        tuple: Tuple of three lists containing the train, val and test sets.
    """
    # check that the sum of the sizes is 1
    assert train_size + val_size + test_size == 1, "The sum of the sizes must be 1"

    # get the number of samples7
    n_samples = len(data_block)

    # get the number of samples for each set
    n_train = int(n_samples * train_size)
    n_val = int(n_samples * val_size)
    n_test = int(n_samples * test_size)

    # get the indices for each set
    indices = np.arange(n_samples)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]

    # get the samples for each set
    train = [data_block[i] for i in train_indices]
    val = [data_block[i] for i in val_indices]
    test = [data_block[i] for i in test_indices]

    return train, val, test


def get_node_features(token_box: dict):
    node_features = [
        v if isinstance(v, Iterable) else [v]
        for k, v in token_box.items()
        if k in NODE_FEATURES
    ]

    assert len(NODE_FEATURES) == len(node_features), (
        "mismatch in the number of node features ",
        f"expected => {len(NODE_FEATURES)} ",
        f"current  => {len(node_features)}",
    )

    node_features = list(flatten(node_features))
    return node_features


def get_labels(datablock) -> Iterator[str]:
    labels = (
        (token["label"] for token in data_item["token_boxes"])
        for data_item in datablock
    )

    labels = flatten(labels)
    return labels


def set_label_map(datablock):
    labels = unique_everseen(get_labels(datablock))
    
    label_map = {
        label : idx
        for idx, label in enumerate(labels)
    }

    inv_label_map = {v: k for k, v in label_map.items()}
    
    return label_map, inv_label_map


def get_doc_graph(data_item, label_map, train_flow) -> nx.DiGraph:
    doc_graph = data_item["doc_graph"]
    if train_flow:
        data_map = {
            token_box["id_line_group"]: {
                "node_features": get_node_features(token_box),
                "label": label_map[token_box["label"]],
            }
            for token_box in data_item["token_boxes"]
        }

        
        node_attributes = {
            node: {
                "x": data_map[node]["node_features"],
                "y": data_map[node]["label"],
            }
            for node in doc_graph.nodes
        }

    else:
        data_map = {
            token_box["id_line_group"]: {
                "node_features": get_node_features(token_box),
            }
            for token_box in data_item["token_boxes"]
        }

        node_attributes = {
            node: {
                "x": data_map[node]["node_features"],
            }
            for node in doc_graph.nodes
        }

    nx.set_node_attributes(doc_graph, node_attributes)
    return doc_graph


def get_pg_graph(doc_graph: nx.DiGraph, train_flow) -> Data:
    if train_flow:
        pg_graph = from_networkx(doc_graph)
        pg_graph.x = pg_graph.x.float()
        pg_graph.y = pg_graph.y.long()
    else:
        pg_graph = from_networkx(doc_graph)
        pg_graph.x = pg_graph.x.float()
    return pg_graph


def get_pg_graphs(data_block, label_map, train_flow) -> list[Data]:
    doc_graphs = [get_doc_graph(data_item,label_map, train_flow) for data_item in data_block]
    pg_graph = [get_pg_graph(doc_graph, train_flow) for doc_graph in doc_graphs]

    return pg_graph