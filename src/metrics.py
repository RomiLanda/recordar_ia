import pandas as pd
from jiwer import measures
from more_itertools import flatten
from shapely.geometry import box as shapely_box


def get_ocr_segments(data_item):

    for idx, segmento in enumerate(data_item['segments']):
        ocr_by_segment = {
                        'id_line_group': [],
                        'text': []
                        }
        for token_box in data_item['token_boxes']:
            if token_box['box_polygon'].intersects(segmento['polygon']):
                ocr_by_segment['id_line_group'].append(token_box['id_line_group'])
                ocr_by_segment['text'].append(token_box['text'])

    df_ocr_by_segment = pd.DataFrame(data=ocr_by_segment)
    data_item['segments'][idx]['ocr_text'] = ' '.join(df_ocr_by_segment.sort_values(by='id_line_group')['text'])

    return data_item

def add_ocr_segments(data_block):
    return [get_ocr_segments(data_item) for data_item in data_block]


def get_metrics(truth: str, hypothesis: str) -> tuple:
    wer = measures.wer(truth, hypothesis)
    mer = measures.mer(truth, hypothesis)
    wil = measures.wil(truth, hypothesis)
    wip = measures.wip(truth, hypothesis)
    cer = measures.cer(truth, hypothesis)

    return wer, mer, wil, wip, cer


def segment_eval(annotations: str, text: str) -> dict:
    metrics = {
        "wer" : 0.0,
        "mer" : 0.0,
        "wil" : 0.0,
        "wip" : 0.0,
        "cer" : 0.0,
    }

    if not annotations or not text:
        return metrics

    wer, mer, wil, wip, cer = get_metrics(annotations, text)

    metrics["wer"] = wer
    metrics["mer"] = mer
    metrics["wil"] = wil
    metrics["cer"] = cer

    return metrics

EVAL_COLUMNS = [
    "label",
    "ocr_metrics",
    "content",
    "ocr_text"
]

def model_evaluation(data_block):
    for data_item in data_block:
        for segmento in data_item['segments']:
            segmento.update(segment_eval(segmento.content, segmento.ocr_text))

    segments_metrics = pd.DataFrame(flatten(data_item['segments'] for data_item in data_block))[EVAL_COLUMNS]

    summary = (
        segments_metrics[["label", "wer", "mer", "wil", "wip", "cer"]]
        .groupby("label")
        .mean()
        )
    
    global_metrics = pd.DataFrame(
        segments_metrics[
            [
                "wer", 
                "mer", 
                "wil",
                "wip",
                "cer"
            ]
        ].mean()
    )

    global_metrics.rename(columns={0: "global_average"}, inplace=True)

    return segments_metrics, summary, global_metrics