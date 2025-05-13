import abc
from abc import abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)

def compute_pr(true_positives: int, false_cases: int, default: float = 0.0) -> float:
    if not (true_positives + false_cases):
        return default
    return true_positives / (true_positives + false_cases)


def compute_f1(precision: float, recall: float, default: float = 0.0) -> float:
    if not (precision + recall):
        return default
    return (2 * precision * recall) / (precision + recall)


@dataclass
class Metric(metaclass=abc.ABCMeta):
    name: str = "Default"
    _keys: Optional[Tuple] = None
    output_dir: Optional[str] = None
    field_name: Optional[str] = None

    @classmethod
    def __subclasshook__(cls, __subclass: type) -> bool:
        methods = ['update', 'compute', 'reset']
        decision = True
        for method in methods:
            decision = decision and (hasattr(__subclass, method) and callable(getattr(__subclass, method)))
        decision = decision and (cls.name != "Default")
        return decision or NotImplemented

    @abstractmethod
    def update(self, reference: any, prediction: any, **kwargs):
        """Update current metric between reference and prediction"""
        raise NotImplementedError

    @abstractmethod
    def compute(self) -> dict:
        """Compute current metric between reference and prediction"""
        raise NotImplementedError

    @abstractmethod
    def reset(self):
        """Reset current metric between reference and prediction"""
        raise NotImplementedError

    def compute_all(self, references: list, predictions: list) -> Dict[str, float]:
        for ref, pred in zip(references, predictions):
            self.update(ref, pred)
        return self.compute()

    def _get_dict(self):
        obj = self.__dict__
        if self._keys:
            obj = {k: v for k, v in obj.items() if k in self._keys}
        return obj

    def __call__(self, reference: any = None, prediction: any = None, compute: bool = False) -> Union[Dict[str, float], None]:
        if reference and prediction:
            self.update(reference, prediction)
        if compute:
            return self.compute()
