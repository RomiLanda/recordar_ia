import pandas as pd
from jiwer import measures
from more_itertools import flatten


def get_metrics(truth: str, hypothesis: str) -> tuple:
    wer = measures.wer(truth, hypothesis)
    mer = measures.mer(truth, hypothesis)
    wil = measures.wil(truth, hypothesis)
    wip = measures.wip(truth, hypothesis)
    cer = measures.cer(truth, hypothesis)

    return wer, mer, wil, wip, cer


def row_eval(annotations: str, text: str) -> dict:
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
    "pred_label",
    "annotations",
    "text",
]

def model_evaluation(data_block):
    flattened_data = flatten(data_item['token_boxes'] for data_item in data_block)
    raw_data = pd.DataFrame(flattened_data)
    report = raw_data[EVAL_COLUMNS].reset_index(drop=True)

    metrics = (
        row_eval(row.annotations, row.text) for _ , row in report.iterrows()
    ) 

    metrics = pd.DataFrame(metrics)
    report = pd.concat([report, metrics], axis=1)

    summary = (
        report[["label", "wer", "mer", "wil", "wip", "cer"]]
        .groupby("label")
        .mean()
        )
    
    global_metrics = pd.DataFrame(
        report[
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

    return report, summary, global_metrics