from __future__ import annotations

from typing import List
from enum import Enum
from jiwer import process_words, process_characters
from transformers import TrOCRProcessor
from evaluate import load

cer_metric = load('cer')
wer_metric = load("wer")
class Metric(Enum):
    WER = "wer"
    CER = "cer"
    
def make_compute(processor: TrOCRProcessor):
    def compute_metrics(pred):
        labels_ids = pred.label_ids
        pred_ids = pred.predictions

        pred_ids[pred_ids == -100] = processor.tokenizer.pad_token_id
        pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
        labels_ids[labels_ids == -100] = processor.tokenizer.pad_token_id
        label_str = processor.batch_decode(labels_ids, skip_special_tokens=True)

        wgt_cer = weighted_score(Metric.CER, label_str, pred_str, weight_factor=0.5)
        wgt_wer = weighted_score(Metric.WER, label_str, pred_str, weight_factor=0.5)
        cer = cer_metric.compute(predictions=pred_str, references=label_str)
        wer = wer_metric.compute(predictions=pred_str, references=label_str)

        return {"wgt_cer": wgt_cer, "wgt_wer": wgt_wer, "cer": cer, "wer": wer}
    return compute_metrics

def weighted_score(metric: Metric, references: List, hypotheses: List, weight_factor: float = 0.5) -> float:
    if metric not in (Metric.WER, Metric.CER):
        raise ValueError("metric must be Metric.WER or Metric.CER")

    weighted_errors = 0.0
    total_weight = 0.0

    for ref, hyp in zip(references, hypotheses):
        if metric == Metric.WER:
            result = process_words(ref, hyp)            
        elif metric == Metric.CER:
            result = process_characters(ref, hyp)

        ref_length = len(result.references[0])
        edits = result.substitutions + result.deletions + result.insertions

        if ref_length == 0:
            continue

        weight = ref_length ** weight_factor

        weighted_errors += edits * weight
        total_weight += weight

    if total_weight == 0:
        return 0.0

    return weighted_errors / total_weight