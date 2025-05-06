from collections import defaultdict
from typing import Callable, List, Dict, Union, Iterable, Any, Optional
from dataclasses import dataclass

from metrics import Metric
from metrics.property_aggregate import PropertyAggregate, PropertyAggregateOrderLevel, GroupedPropertyAggregate, GroupedPropertyAggregateOrderLevel
from metrics.strict import Strict
from metrics.match import Match
from metrics.rouge1 import Rouge1, Rouge1EncounterLevel
from metrics.multilabel import MultiLabel

METRIC_CLS = [Strict, Match, Rouge1, MultiLabel, PropertyAggregate, PropertyAggregateOrderLevel, Rouge1EncounterLevel, GroupedPropertyAggregate, GroupedPropertyAggregateOrderLevel]
METRICS = {m.name: m for m in METRIC_CLS}


@dataclass
class MetricDict:
    metrics: List[Union[Metric, str]]
    parameters: Optional[Dict[str, Dict[str, Any]]] = None
    name: str = "default"
    processor: Optional[Callable] = None
    output_dir: Optional[str] = None

    def __post_init__(self):
        if len(self.metrics) > 0 and isinstance(self.metrics[0], str):
            new_metrics = []
            for metric in self.metrics:

                if metric not in METRICS:
                    raise ValueError(f"Metric name not in available metrics: {list(METRICS.keys())}")
                metric_cls = METRICS.get(metric)
                params = {}
                if self.parameters and metric in self.parameters:
                    params = self.parameters.get(metric, {})
                
                params["output_dir"] = self.output_dir
                params["field_name"] = self.name

                new_metrics.append(metric_cls(**params))
            self.metrics = new_metrics

    def __getitem__(self, key):
        if key not in self.metrics:
            raise KeyError("Key not in metrics.")
        return self.metrics.get(key)

    def update(self, reference: any, prediction: any, preprocessor=None):
        if not preprocessor:
            preprocessor = self.processor
        for metric in self.metrics:
            metric.update(reference, prediction, processor=preprocessor)

    def compute(self) -> Dict[str, float]:
        output = {}
        for metric in self.metrics:
            curr_output = metric.compute()
            curr_output = {f"{metric.name}_{k}": v for k, v in curr_output.items()}
            output.update(curr_output)
        return output

    def reset(self):
        for metric in self.metrics:
            metric.reset()

    def compute_all(
        self, references: Iterable, predictions: Iterable, indices: Optional[Iterable[int]] = None, preprocessor=None
    ) -> Dict[str, float]:

        if indices is not None:
            if len(references) != len(predictions) or len(references) != len(indices):
                raise ValueError("Lengths of references, predictions, and indices must match.")

            # re-create encounters
            ref_encounter = defaultdict(list)
            hyp_encounter = defaultdict(list)
            for ref, hyp, idx in zip(references, predictions, indices):
                ref_encounter[idx].append(ref)
                hyp_encounter[idx].append(hyp)

            for idx in ref_encounter:
                self.update(ref_encounter[idx], hyp_encounter[idx], preprocessor=preprocessor)
        else:
            for ref, pred in zip(references, predictions):
                self.update(ref, pred, preprocessor=preprocessor)
        return self.compute()
