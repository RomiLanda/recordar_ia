import numpy as np
import networkx as nx
from copy import deepcopy
from itertools import chain
from shapely.strtree import STRtree
from shapely import distance, box, centroid
from more_itertools import flatten, windowed
from .debug import doc_debug


def box_buffer(box_polygon, scale=(0,0)):
    """
    Scale = (0,0) returns box without any buffer.
    """
    x1, y1, x2, y2 = box_polygon.bounds
    x_scale, y_scale = scale
    box_buffered = box(x1-((x_scale*(x2-x1))/2), y1-((y_scale*(y2-y1))/2), x2+((x_scale*(x2-x1))/2), y2+((y_scale*(y2-y1))/2))
    return box_buffered


def get_intersections(data_item, dilate_scale=(0,0)):
    all_boxes = [box['box_polygon'] for box in data_item['token_boxes']]
    all_ids = [box['id'] for box in data_item['token_boxes']]
    tree = STRtree(all_boxes)
    edges_i = []
    for i, p in enumerate(all_boxes):
        intersections = tree.query(box_buffer(p, dilate_scale), predicate='intersects')
        intersections = intersections[intersections != i]
        if len(intersections) == 0:
            intersections = np.append(intersections, tree.query_nearest(p, exclusive=True))
        for i_p in intersections:
            edges_i.append((i, i_p, distance(centroid(p), centroid(all_boxes[i_p]))))

    edges_hash = []
    for a, b, w in edges_i:
        edges_hash.append((all_ids[a], all_ids[b], w))

    return edges_hash


def create_doc_graphs(
    data_item,
    debug: bool = False,
    bidirectional: bool = True,
):
    
    data_item = deepcopy(data_item)
    token_boxes = data_item["token_boxes"]

    nodes = (token_box["id"] for token_box in token_boxes)
    rels = get_intersections(data_item, dilate_scale=(0.4, 1))

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