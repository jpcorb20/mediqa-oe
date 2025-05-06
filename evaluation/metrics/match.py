from dataclasses import dataclass
from typing import Dict, Tuple

from metrics import Metric, compute_pr, compute_f1


@dataclass
class Match(Metric):
    name: str = "Match"
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    export_counts: bool = False
    _keys: Tuple = ("true_positives", "false_positives", "false_negatives")

    def update(self, reference: any, prediction: any, **kwargs):

        if reference or prediction:
            if not reference:
                self.false_positives += 1
                return
            elif not prediction:
                self.false_negatives += 1
                return
            
        if reference and prediction:
            self.true_positives += 1
            
    def compute(self) -> Dict[str, float]:
        output = self._get_dict() if self.export_counts else {}
        output["precision"] = compute_pr(self.true_positives, self.false_positives)
        output["recall"] = compute_pr(self.true_positives, self.false_negatives)
        output["f1"] = compute_f1(output["precision"], output["recall"])
        return output

    def reset(self):
        self.true_positives = 0
        self.false_positives = 0
        self.false_negatives = 0
