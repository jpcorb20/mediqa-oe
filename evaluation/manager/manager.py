import os
import json
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Generator, Union

from preprocessing import PreprocessorConfig, Preprocessor
from metrics.dict import MetricDict
from utils.slice import slice_gen


@dataclass
class EvaluationManager:
    output_directory: str
    fields: Dict[str, MetricDict]
    preprocessings: Dict[str, bool]
    preprocessor_config: Union[PreprocessorConfig, None] = None
    preprocessor_config_path: str = ""
    latest_output: Union[Dict[str, Dict[str, float]], None] = None
    orders_metrics: MetricDict = None
    encounter_metrics: MetricDict = None  

    def __post_init__(self):
        if self.preprocessor_config_path:
            self.preprocessor = Preprocessor.from_json_path(self.preprocessor_config_path)
        elif self.preprocessor_config:
            self.preprocessor = Preprocessor.from_config(self.preprocessor_config)
        else:
            self.preprocessor = None
        
        if self.preprocessor and self.preprocessings["order_level_metrics"]:
            self.orders_metrics.processor = self.preprocessor
        if self.preprocessor and self.preprocessings["encounter_level_metrics"]:
            self.encounter_metrics.processor = self.preprocessor

    def _prepare_from_dicts(self, items: List[Dict[str, any]], field: str) -> Generator:
        if self.preprocessor and self.preprocessings.get(field):
            out_gen = slice_gen(items, field, self.preprocessor)
        else:
            out_gen = slice_gen(items, field)
        return out_gen

    def process(self, references: List[Dict[str, any]], predictions: List[Dict[str, any]], indices: List[int]) -> Dict[str, Dict[str, float]]:
        output = {}
        for k, m in self.fields.items():
            refs = self._prepare_from_dicts(references, k)
            preds = self._prepare_from_dicts(predictions, k)
            output[k] = m.compute_all(refs, preds)

        # Order level metrics
        output["order_level_metrics"] = self.orders_metrics.compute_all(references, predictions, preprocessor=self.preprocessor)
        output["encounter_level_metrics"] = self.encounter_metrics.compute_all(references, predictions, indices, preprocessor=self.preprocessor)
        self.latest_output = output
        return output

    def export(self, filename: str = ""):
        if self.output_directory:
            if not filename:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = "results" + timestamp
            output_path = os.path.join(self.output_directory, filename + ".json")
            with open(output_path, "w") as fp:
                json.dump(self.latest_output, fp)

    @classmethod
    def from_paths(cls, path: str, output_directory: str) -> "EvaluationManager":
        """Load manager with path to config and output directory path."""
        with open(path, "r") as fp:
            config = json.load(fp)
        return cls.from_dict(config, output_directory)

    @classmethod
    def from_dict(cls, config: Dict[str, any], output_directory: str) -> "EvaluationManager":
        preprocess_config = PreprocessorConfig.from_json(config.pop("preprocessor_config"))

        order_level_metrics = config.pop("order_level_metrics", {})
        encounter_level_metrics = config.pop("encounter_level_metrics", {})

        # Those metrics are computed on the whole order, not on individual fields
        orders_metrics = encounters_metrics = None
        if order_level_metrics:
            orders_metrics = MetricDict(name="order_level", 
                                        metrics=order_level_metrics["metrics"], 
                                        parameters=order_level_metrics.get("parameters"),
                                        output_dir=output_directory)
        if encounter_level_metrics:
            encounters_metrics = MetricDict(name="encounter_level", 
                                            metrics=encounter_level_metrics["metrics"], 
                                            parameters=encounter_level_metrics.get("parameters"),
                                            output_dir=output_directory)

        fields = {c: MetricDict(name=c, metrics=m["metrics"], parameters=m.get("parameters"), output_dir=output_directory) for c, m in config.items()}
        processings = {c: m["preprocess"] for c, m in config.items()}
        
        processings["order_level_metrics"] = order_level_metrics["preprocess"]
        processings["encounter_level_metrics"] = encounter_level_metrics["preprocess"]

        return cls(output_directory, fields, processings, 
                   preprocessor_config=preprocess_config, 
                   orders_metrics=orders_metrics, 
                   encounter_metrics=encounters_metrics)