import os
from datetime import datetime
from typing import List, Dict, Tuple, Union, Generator
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from preprocessing import Preprocessor, PreprocessorConfig
from utils.slice import slice_gen
from order import Order
from pairing.list_manipulators import *


@dataclass
class PairingMatcher:
    output_directory: str
    preprocessing_config: PreprocessorConfig
    field: str = "description"
    preprocessing: Union[Preprocessor, None] = None
    pairings_accumulator: Union[List[Tuple[str, str, float]], None] = None
    encounter_index: int = 0

    def __post_init__(self):
        if not self.preprocessing:
            self.preprocessing = Preprocessor.from_config(self.preprocessing_config)
        self.accumulator_reset()
        self.encounter_index = 0

    def accumulator_reset(self):
        self.pairings_accumulator = []

    def _prepare_from_dicts(self, items: List[Dict[str, any]], field: str) -> Generator:
        if self.preprocessing:
            out_gen = slice_gen(items, field, self.preprocessing)
        else:
            out_gen = slice_gen(items, field)
        return out_gen

    def pairing_metric(self, truth: str, pred: str) -> float:
        match = 0.0
        if truth == pred:
            match = 1.0
        else:
            pred_words = pred.split()
            truth_words = truth.split()
            local = 0
            if len(truth_words):
                for word in truth_words:
                    if word in pred_words:
                        local += 1
                match = (local / len(truth_words))
        return match

    def build_metric_matrix(self, ref: Generator[str, None, None], hyp: Generator[str, None, None]) -> np.array:
        matrix = []
        local_hyp = list(hyp) # Generator needs to be used len(ref) times.
        for order1 in ref:
            row = []
            for order2 in local_hyp:
                row.append(self.pairing_metric(order1, order2))
            matrix.append(row)

        # Handle edge case where matrix is empty (i.e. matrix should be 2D).
        if not matrix:
            matrix = [[]]

        return np.array(matrix)

    def pair(
        self,
        ref: List[Dict[str, Union[str, int]]],
        hyp: List[Dict[str, Union[str, int]]],
    ) -> Tuple[List[List[Union[Dict[str, Union[str, int]], None]]], List[float]]:
        ref_gen = self._prepare_from_dicts(ref, self.field)
        hyp_gen = self._prepare_from_dicts(hyp, self.field)
        cost_matrix = self.build_metric_matrix(ref_gen, hyp_gen)
        row_ind, col_ind = linear_sum_assignment(cost_matrix, maximize=True)

        # Make sure cost is above zero else pop out (no actual pair)
        nonzero_costs = cost_matrix[row_ind, col_ind] > 0
        if not np.all(nonzero_costs):
            nonzero_indices = np.where(nonzero_costs)[0]
            row_ind = row_ind[nonzero_indices]
            col_ind = col_ind[nonzero_indices]
        scores = cost_matrix[row_ind, col_ind].tolist()

        # Match pairs
        row_ind, col_ind = row_ind.tolist(), col_ind.tolist()
        pairings = (slice_items(ref, row_ind), slice_items(hyp, col_ind))  # (N, 2)
        pairings = nest_tup_to_nest_list(zip(*pairings))  # transpose (2, N)

        # Add missing elements with None.
        if len(ref) > len(hyp) or len(ref) > len(row_ind):
            miss_pairs = get_miss_items(ref, row_ind)
            scores.extend([0.0] * len(miss_pairs))
            pairings.extend(miss_pairs)
        if len(ref) < len(hyp) or len(hyp) > len(col_ind):
            miss_pairs = get_miss_items(hyp, col_ind, none_side="left")
            scores.extend([0.0] * len(miss_pairs))
            pairings.extend(miss_pairs)

        for (p1, p2), s in zip(pairings, scores):
            self.pairings_accumulator.append(dict(ref=p1, hyp=p2, score=s, index=self.encounter_index))

        return pairings, scores

    def get_pairings(self, transpose: bool = False):
        output = [[p.get("ref"), p.get("hyp"), p.get("index")] for p in self.pairings_accumulator]
        if transpose:
            output = list(zip(*output))

        if output == []:
            output = [[], [], []]
        return output

    def get_pairings_accumulator(self):
        return self.pairings_accumulator

    def pairings_to_tsv(self) -> str:
        doc = []
        header = ["transcript_id"]
        pair_keys = set(Order.__annotations__.keys())
        for i, p in enumerate(self.pairings_accumulator):
            # Parse transcript id.
            transcript_id = (p["ref"] or p["hyp"]).get("transcript_id")
            if p["ref"]:
                del p["ref"]["transcript_id"]
            if p["hyp"]:
                del p["hyp"]["transcript_id"]
            line = [transcript_id]

            # Parse field next to each other.
            for field in pair_keys:
                for k in {"ref", "hyp"}:
                    if i == 0:
                        header.append(f"{k}_{field}")
                    v = p.get(k)
                    value = v.get(field, "") if v else ""
                    write_val = str(value)
                    if isinstance(value, list) or isinstance(value, str):
                        write_val = f'"{write_val}"'
                    line.append(write_val)

            # Parse score.
            if i == 0:
                header.append("score")
            line.append(str(p["score"]))

            doc.append(line)
        doc = [header] + doc
        output = "\n".join("\t".join(line) for line in doc)
        return output

    def export(self, filename: str = ""):
        if self.output_directory:
            if not filename:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"pairings_{timestamp}"
            output_path = os.path.join(self.output_directory, filename + ".tsv")
            with open(output_path, "w") as fp:
                fp.write(self.pairings_to_tsv())

    def __call__(
        self,
        ref: List[Dict[str, Union[str, int]]],
        hyp: List[Dict[str, Union[str, int]]]
    ) -> Tuple[List[List[Union[Dict[str, Union[str, int]], None]]], List[float]]:
        self.encounter_index += 1
        return self.pair(ref, hyp)
