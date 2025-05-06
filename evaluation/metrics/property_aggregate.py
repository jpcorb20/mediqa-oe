import os
from dataclasses import dataclass, field
from typing import Dict
import matplotlib.pyplot as plt
from collections import defaultdict

from metrics import Metric


def score_property(ref, hyp, processor=None):
    if not ref or not hyp:
        return 0, 0

    if processor:
        ref = processor(ref)
        hyp = processor(hyp)

    hyp = hyp.strip()
    hyp = hyp.lower()
    hyp_words = hyp.split()
    hyp_words = set(hyp.split())

    ref = ref.strip()
    ref = ref.lower()
    ref_words = ref.split()
    ref_words = set(ref.split())

    p_correct = r_correct = 0
    nb_relevants = len(ref_words)
    nb_retrieved = len(hyp_words)

    for word in hyp_words:
        if word in ref_words:
            p_correct += 1

    for word in ref_words:
        if word in hyp_words:
            r_correct += 1

    precision = recall = 0
    if nb_retrieved == 0:
        precision = 0
    else:
        precision = p_correct / nb_retrieved

    if nb_relevants == 0:
        recall = 0
    else:
        recall = r_correct / nb_relevants

    return precision, recall


def plot_bars(bar_plot_savepath, output, precision, include, mode):
        if bar_plot_savepath is not None:
            os.makedirs(os.path.dirname(bar_plot_savepath), exist_ok=True)
            # create bar plot of f1, precision and recall for each group
            xlabels = sorted(list(precision.keys()))
            groupstrs = [f"{group}_" if len(group) > 0 else "" for group in xlabels]
            f1s = [output[f"{group}f1"] for group in groupstrs]
            precisions = [output[f"{group}precision"] for group in groupstrs]
            recalls = [output[f"{group}recall"] for group in groupstrs]

            x = range(len(xlabels))
            width = 0.2

            _, ax = plt.subplots(figsize=(10, 6))
            ax.bar(x, precisions, width, label="Precision")
            ax.bar([p + width for p in x], recalls, width, label="Recall")
            ax.bar([p + width * 2 for p in x], f1s, width, label="F1 Score")

            ax.set_xlabel("Groups")
            ax.set_ylabel("Scores")
            title_label = include[0] if include and len(include) == 1 else "Properties"
            ax.set_title(f"Precision, Recall, and F1 Score of {title_label} by Group ({mode})")
            ax.set_xticks([p + width for p in x])
            ax.set_xticklabels(xlabels, rotation=45)
            ax.legend()

            plt.tight_layout()
            plt.savefig(bar_plot_savepath + f"_{mode}.png")
            plt.close()


@dataclass
class GroupedPropertyAggregate(Metric):
    name: str = "grouped_property_aggregate"
    include: list[str] = None  # if include specified, then only those are used
    exclude: list[str] = None  # if include not specified, then all except exclude are used
    precision: Dict[str, float] = field(default_factory=dict)
    recall: Dict[str, float] = field(default_factory=dict)
    ref_retrieved: Dict[str, int] = field(default_factory=dict)
    hyp_retrieved: Dict[str, int] = field(default_factory=dict)
    group_by_property: str = None
    create_bar_plot: bool = True

    def _init_group(self, group):
        if group not in self.precision:
            self.precision[group] = 0.0
            self.recall[group] = 0.0
            self.ref_retrieved[group] = 0
            self.hyp_retrieved[group] = 0

    def _get_group(self, order, processor):
        if self.group_by_property:
            if processor:
                return processor(order[self.group_by_property])
            return order[self.group_by_property]
        return ""

    def update(self, reference: any, prediction: any, processor=None, **kwargs):

        if prediction:
            group = self._get_group(prediction, processor)
            self._init_group(group)

            considered_attributes = self.include if self.include else [k for k in prediction.keys() if k not in self.exclude]
            for attr in considered_attributes:
                if prediction[attr]:
                    self.hyp_retrieved[group] += 1

        if reference:
            group = self._get_group(reference, processor)
            self._init_group(group)

            considered_attributes = self.include if self.include else [k for k in reference.keys() if k not in self.exclude]
            for attr in considered_attributes:
                if reference[attr]:
                    self.ref_retrieved[group] += 1

                # comment: computing this only here, means that all orders where the model predicted
                # garbage properties that are not part of the reference, will not factor into the score.
                if prediction and (attr in prediction):
                    precision, recall = score_property(reference[attr], prediction[attr], processor)
                    self.precision[group] += precision
                    self.recall[group] += recall


    def compute(self) -> Dict[str, float]:
        output = {}
        for group in self.precision:
            groupstr = f"{group}_" if len(group) > 0 else ""
            output[f"{groupstr}precision"] = (
                0 if not self.hyp_retrieved[group] else self.precision[group] / self.hyp_retrieved[group]
            )
            output[f"{groupstr}recall"] = (
                0 if not self.ref_retrieved[group] else self.recall[group] / self.ref_retrieved[group]
            )

            if output[f"{groupstr}precision"] + output[f"{groupstr}recall"] == 0:
                output[f"{groupstr}f1"] = 0
            else:
                output[f"{groupstr}f1"] = (
                    2
                    * (output[f"{groupstr}precision"] * output[f"{groupstr}recall"])
                    / (output[f"{groupstr}precision"] + output[f"{groupstr}recall"])
                )

        if self.output_dir and self.create_bar_plot:
            groupbystr = self.group_by_property if self.group_by_property else "global"
            bar_plot_savepath = os.path.join(
                self.output_dir,
                self.field_name + f"_{groupbystr}_bar_plot"
                )
            plot_bars(bar_plot_savepath, output, self.precision, self.include, mode="micro")

        return output

    def reset(self):
        self.precision = {}
        self.recall = {}
        self.ref_retrieved = {}
        self.hyp_retrieved = {}


@dataclass
class PropertyAggregate(GroupedPropertyAggregate):
    name: str = "property_aggregate"
    create_bar_plot: bool = False
    
    def __post_init__(self):
        # group by property may must be None
        if self.group_by_property:
            raise ValueError("PropertyAggregate requires group_by_property to be None")
        self._init_group("")


@dataclass
class GroupedPropertyAggregateOrderLevel(Metric):
    name: str = "grouped_property_aggregate_order_level"
    include: list[str] = None  # if include specified, then only those are used
    exclude: list[str] = None  # if include not specified, then all except exclude are used
    precision: Dict[str, float] = field(default_factory=dict)
    recall: Dict[str, float] = field(default_factory=dict)
    ref_retrieved: Dict[str, int] = field(default_factory=dict)
    hyp_retrieved: Dict[str, int] = field(default_factory=dict)
    group_by_property: str = None
    create_bar_plot: bool = True

    def _init_group(self, group):
        if group not in self.precision:
            self.precision[group] = 0.0
            self.recall[group] = 0.0
            self.ref_retrieved[group] = 0
            self.hyp_retrieved[group] = 0

    def _get_group(self, order, processor):
        if self.group_by_property:
            if processor:
                return processor(order[self.group_by_property])
            return order[self.group_by_property]
        return ""

    def update(self, reference: any, prediction: any, processor=None, **kwargs):

        ref_prop_retrieved = defaultdict(int)
        hyp_prop_retrieved = defaultdict(int)

        inner_recall = defaultdict(float)
        inner_precision = defaultdict(float)

        if prediction:
            group = self._get_group(prediction, processor)
            self._init_group(group)

            considered_attributes = self.include if self.include else [k for k in prediction.keys() if k not in self.exclude]
            for attr in considered_attributes:
                if prediction[attr]:
                    hyp_prop_retrieved[group] += 1

        if reference:
            group = self._get_group(reference, processor)
            self._init_group(group)

            considered_attributes = self.include if self.include else [k for k in reference.keys() if k not in self.exclude]
            for attr in considered_attributes:
                if reference[attr]:
                    ref_prop_retrieved[group] += 1

                # comment: computing this only here, means that all orders where the model predicted
                # garbage properties that are not part of the reference, will not factor into the score.
                if prediction and (attr in prediction):
                    precision, recall = score_property(reference[attr], prediction[attr], processor)
                    inner_precision[group] += precision
                    inner_recall[group] += recall

        for group in self.precision:
            self.hyp_retrieved[group] += 1 if hyp_prop_retrieved[group] else 0
            self.ref_retrieved[group] += 1 if ref_prop_retrieved[group] else 0
            self.recall[group] += inner_recall[group] / ref_prop_retrieved[group] if ref_prop_retrieved[group] else 0
            self.precision[group] += inner_precision[group] / hyp_prop_retrieved[group] if hyp_prop_retrieved[group] else 0


    def compute(self) -> Dict[str, float]:
        output = {}
        for group in self.precision:
            groupstr = f"{group}_" if len(group) > 0 else ""
            output[f"{groupstr}precision"] = (
                0 if not self.hyp_retrieved[group] else self.precision[group] / self.hyp_retrieved[group]
            )
            output[f"{groupstr}recall"] = (
                0 if not self.ref_retrieved[group] else self.recall[group] / self.ref_retrieved[group]
            )

            if output[f"{groupstr}precision"] + output[f"{groupstr}recall"] == 0:
                output[f"{groupstr}f1"] = 0
            else:
                output[f"{groupstr}f1"] = (
                    2
                    * (output[f"{groupstr}precision"] * output[f"{groupstr}recall"])
                    / (output[f"{groupstr}precision"] + output[f"{groupstr}recall"])
                )

        if self.output_dir and self.create_bar_plot:
            groupbystr = self.group_by_property if self.group_by_property else "global"
            bar_plot_savepath = os.path.join(
                self.output_dir,
                self.field_name + f"_{groupbystr}_bar_plot"
                )
            plot_bars(bar_plot_savepath, output, self.precision, self.include, mode="macro")

        return output

    def reset(self):
        self.precision = {}
        self.recall = {}
        self.ref_retrieved = {}
        self.hyp_retrieved = {}


@dataclass
class PropertyAggregateOrderLevel(GroupedPropertyAggregateOrderLevel):
    name: str = "property_aggregate_order_level"
    create_bar_plot: bool = False
    
    def __post_init__(self):
        # group by property may must be None
        if self.group_by_property:
            raise ValueError("PropertyAggregateOrderLevel requires group_by_property to be None")
        self._init_group("")