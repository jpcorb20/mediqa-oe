from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List

from metrics import Metric, compute_f1

def process_text(text: any, processor = None) -> List[str]:
    if isinstance(text, str):
        if processor:
            text = processor(text)

        text = text.strip()
        text = text.lower()
        words = text.split()
    else:
        words = str(text).split()
    return words

@dataclass
class Rouge1(Metric):
    name: str = "Rouge1"
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

        pred_words = process_text(prediction)

        ref_words = process_text(reference)

        if prediction:
            nb_retrieved = len(pred_words)
            self.sum_nb_retrieved += 1

        if reference:
            nb_relevants = len(ref_words)
            self.sum_nb_relevants += 1

        for word in ref_words:
            if word in pred_words:
                recall_nb_correct += 1

        for word in pred_words:
            if word in ref_words:
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


@dataclass
class Rouge1EncounterLevel(Metric):
    name: str = "Rouge1_encounter_level"
    properties: List[str] = None
    property_values: defaultdict = None

    def __post_init__(self):
        self.properties = self.properties or []
        self.property_values = defaultdict(lambda: defaultdict(float))

    def update(self, references: any, predictions: any, processor = None, **kwargs):

        for prop in self.properties:
            self.property_values[prop]["num_encounter"] += 1

        num_order_hyp = 0
        num_order_ref = 0
        encounter_values = defaultdict(lambda: defaultdict(float))

        for ref, pred in zip(references, predictions):

            recall_nb_correct = 0
            precision_nb_correct = 0
            nb_retrieved = 0
            nb_relevants = 0
            for prop in self.properties:

                reference = ref.get(prop, "") if ref else ""
                prediction = pred.get(prop, "") if pred else ""

                if not reference and not prediction:
                    continue

                pred_words = process_text(prediction, processor)
                ref_words = process_text(reference, processor)

                if prediction:
                    self.property_values[prop]["retrieved"] += 1
                    nb_retrieved = len(pred_words)

                if reference:
                    self.property_values[prop]["relevants"] += 1
                    nb_relevants = len(ref_words)

                for word in ref_words:
                    if word in pred_words:
                        recall_nb_correct += 1

                for word in pred_words:
                    if word in ref_words:
                        precision_nb_correct += 1

                precision = 0.0
                if nb_retrieved > 0:
                    num_order_hyp += 1
                    precision = precision_nb_correct / nb_retrieved

                recall = 0.0
                if nb_relevants > 0:
                    num_order_ref += 1
                    recall = recall_nb_correct / nb_relevants

                encounter_values[prop]["precision"] += precision
                encounter_values[prop]["recall"] += recall

        for prop in encounter_values:
            self.property_values[prop]["sum_precision"] += (
                encounter_values[prop]["precision"] / num_order_hyp
                if num_order_hyp > 0
                else 0
            )
            self.property_values[prop]["sum_recall"] += (
                encounter_values[prop]["recall"] / num_order_ref
                if num_order_ref > 0
                else 0
            )

    def compute(self) -> Dict[str, float]:
        output = {}

        for prop in self.properties:
            output[prop] = {}
            output[prop]["precision"] = (
                self.property_values[prop]["sum_precision"]
                / self.property_values[prop]["num_encounter"]
                if self.property_values[prop]["num_encounter"] > 0
                else 0
            )
            output[prop]["recall"] = (
                self.property_values[prop]["sum_recall"]
                / self.property_values[prop]["num_encounter"]
                if self.property_values[prop]["num_encounter"] > 0
                else 0
            )
            output[prop]["f1"] = compute_f1(
                output[prop]["precision"], output[prop]["recall"]
            )

        return output

    def reset(self):
        self.sum_precision = 0.0
        self.sum_recall = 0.0
        self.sum_nb_retrieved = 0
        self.sum_nb_relevants = 0
        self.property_values = defaultdict(lambda: defaultdict(float))
