from dataclasses import dataclass
from typing import Dict, Tuple

from metrics import Metric, compute_f1


@dataclass
class Strict(Metric):
    name: str = "Strict"
    true_positives: int = 0
    nb_retrieved: int = 0
    nb_relevants: int = 0
    export_counts: bool = False
    _keys: Tuple = ("true_positives", "nb_retrieved", "nb_relevants")

    def update(self, reference: any, prediction: any, **kwargs):
        if reference:
            self.nb_relevants += 1

        if prediction:
            self.nb_retrieved += 1

        if not prediction and not reference:
            return

        if reference == prediction:
            self.true_positives += 1

    def compute(self) -> Dict[str, float]:
        output = self._get_dict() if self.export_counts else {}
        output.update({"precision": 0.0, "recall": 0.0})
        if self.nb_retrieved > 0:
            output["precision"] = self.true_positives / self.nb_retrieved
        if self.nb_relevants > 0:
            output["recall"] = self.true_positives / self.nb_relevants
        output["f1"] = compute_f1(output["precision"], output["recall"])
        return output

    def reset(self):
        self.true_positives = 0
        self.false_positives = 0
        self.false_negatives = 0
