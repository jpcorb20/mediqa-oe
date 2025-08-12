from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List

from metrics import Metric, compute_f1

def process_list(obj: any) -> List[int]:
    if isinstance(obj, list):
        return [int(e) for e in obj if isinstance(e, int) or (isinstance(e, str) and e.isnumeric())]
    elif isinstance(obj, str):
        if obj.strip() == "":
            return []
        elif obj.startswith("[") and obj.endswith("]"):
            return [int(e) for e in obj[1:-1].split(",")]
        elif obj.isnumeric():
            return [int(obj)]
    else:
        print(f"Warning: {obj} is not a list but {type(obj)}")
        raise ValueError(f"Unsupported type: {type(obj)}")

@dataclass
class MultiLabel(Metric):
    name: str = "MultiLabel"
    sum_precision: float = 0
    sum_recall: float = 0
    sum_nb_retrieved: int = 0
    sum_nb_relevants: int = 0

    def update(self, reference: any, prediction: any, **kwargs):
        recall_nb_correct = 0
        precision_nb_correct = 0
        nb_retrieved = 0
        nb_relevants = 0

        if not reference and not prediction:
            return

        pred_labels = process_list(prediction)
        ref_labels = process_list(reference)

        print(f"Reference: {reference}, Prediction: {prediction}")
        print(f"Processed Reference: {ref_labels}, Processed Prediction: {pred_labels}")

        if prediction:
            nb_retrieved = len(pred_labels)
            self.sum_nb_retrieved += 1

        if reference:
            nb_relevants = len(ref_labels)
            self.sum_nb_relevants += 1

        for label in ref_labels:
            if label in pred_labels:
                recall_nb_correct += 1

        for label in pred_labels:
            if label in ref_labels:
                precision_nb_correct += 1

        precision = 0.0
        if nb_retrieved > 0:
            precision = precision_nb_correct / nb_retrieved

        recall = 0.0
        if nb_relevants > 0:
            recall = recall_nb_correct / nb_relevants

        self.sum_precision += precision
        self.sum_recall += recall

    def compute(self) -> Dict[str, float]:
        output = {"precision": 0.0, "recall": 0.0}
        if self.sum_nb_retrieved > 0:
            output["precision"] = self.sum_precision / self.sum_nb_retrieved
        if self.sum_nb_relevants > 0:
            output["recall"] = self.sum_recall / self.sum_nb_relevants
        output["f1"] = compute_f1(output["precision"], output["recall"])
        return output

    def reset(self):
        self.sum_precision = 0.0
        self.sum_recall = 0.0
        self.sum_nb_retrieved = 0
        self.sum_nb_relevants = 0

