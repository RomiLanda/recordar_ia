import networkx as nx
from copy import deepcopy
from itertools import chain
from src.ocr_boxes import get_line_groups
from more_itertools import flatten, windowed

from .debug import doc_debug
from .utils import get_line_center, get_boxes_ditance


V_REL_WINDOW = 5
MAX_V_WEIGHT = 0.25


def get_v_rels(token_boxes: list[dict], image_height: int):
    line_groups = get_line_groups(token_boxes)
    line_groups = list(line_groups)

    for idx, line_group in enumerate(line_groups):
        for top_token in line_group:
            top_box = top_token["box"]

            bottom_idx_start = idx + 1
            bottom_tokens = flatten(line_groups[bottom_idx_start:])
            candidate_bottom_tokens = []
            for bottom_token in bottom_tokens:
                bottom_box = bottom_token["box"]

                top_x_range = range(top_box[0], top_box[2] + 1)
                bottom_x_range = range(bottom_box[0], bottom_box[2] + 1)

                if not set(top_x_range).intersection(set(bottom_x_range)):
                    continue

                candidate_box = {
                    "n_line": bottom_token["n_line"],
                    "bottom_token": bottom_token,
                }

                candidate_bottom_tokens.append(candidate_box)

            if not candidate_bottom_tokens:
                continue

            min_line = min(map(lambda x: x["n_line"], candidate_bottom_tokens))
            if (min_line - top_token["n_line"]) > V_REL_WINDOW:
                continue

            valid_bottom_tokens = filter(
                lambda x: x["n_line"] == min_line, candidate_bottom_tokens
            )

            valid_bottom_tokens = map(
                lambda x: x["bottom_token"], valid_bottom_tokens
            )

            for valid_bottom_token in valid_bottom_tokens:
                bottom_box = valid_bottom_token["box"]
                boxes_ditance = get_boxes_ditance(top_box, bottom_box)
                if not boxes_ditance:
                    continue
                    
                if (boxes_ditance / image_height) > MAX_V_WEIGHT:
                    continue

                rel = (
                    top_token["id"],
                    valid_bottom_token["id"],
                    boxes_ditance,
                )

                yield rel


def get_h_rels(token_boxes: list[dict]):
    line_groups = get_line_groups(token_boxes)
    for tokens in line_groups:
        for left_token, rigth_token in windowed(tokens, 2):

            if not left_token or not rigth_token:
                continue

            left_box = left_token["box"]
            rigth_box = rigth_token["box"]
            boxes_ditance = get_boxes_ditance(left_box, rigth_box)
            if not boxes_ditance:
                continue

            rel = (left_token["id"], rigth_token["id"], boxes_ditance)
            yield rel


def create_doc_graphs(
    data_item,
    debug: bool = False,
    bidirectional: bool = True,
):

    data_item = deepcopy(data_item)
    image_height = data_item["image_shape"]["image_height"]
    token_boxes = data_item["token_boxes"]

    nodes = (token_box["id"] for token_box in token_boxes)
    v_rels = get_v_rels(token_boxes, image_height)
    h_rels = get_h_rels(token_boxes)
    rels = list(chain(v_rels, h_rels))

    min_weight = min(weight for *_, weight in rels)
    rels = ((src, tgt, min_weight / weight) for src, tgt, weight in rels)

    doc_graph = nx.DiGraph()
    doc_graph.add_nodes_from(nodes)
    doc_graph.add_weighted_edges_from(rels)

    if bidirectional:
        doc_graph = nx.compose(doc_graph, doc_graph.reverse())

    data_item["doc_graph"] = doc_graph

    if debug:
        doc_debug(data_item)

    return data_item