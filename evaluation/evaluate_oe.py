#!/usr/bin/env python
import os
import json
import argparse
import logging
from typing import List, Dict, Any, Optional, Tuple, Union

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from order import Order
from pairing import PairingMatcher
from manager import EvaluationManager
from preprocessing import PreprocessorConfig
from metrics.dict import MetricDict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


VALID_ORDER_TYPES = {"medication", "lab", "followup", "imaging"}
VALID_ATTRIBUTES = set(Order.__annotations__.keys())

def process_order(obj: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None) -> Tuple[Optional[Dict[str, Any]], bool, bool]:
    """
    Process and normalize order object.

    Args:
        obj: The order object to process
        metadata: Optional metadata to update the order with

    Returns:
        Tuple containing:
        - Processed order (or None if it should be skipped)
        - Boolean indicating if transcript should be skipped (break)
        - Boolean indicating if this specific order should be skipped (continue)
    """
    # Skip if required fields are missing
    if "description" not in obj or not obj["description"]:
        return None, False, True

    # Skip if order type is not in our focus set
    order_type = obj.get("order_type", "").lower()
    if order_type not in VALID_ORDER_TYPES:
        return None, False, True

    # Remove all attributes except the ones we care about
    obj = {k: v for k, v in obj.items() if k in VALID_ATTRIBUTES and v}

    # Add metadata if provided
    if metadata is not None:
        obj.update(metadata)

    return obj, False, False


def process_multiple_orders(order_list: List[Dict[str, Any]], metadata_list: List[Dict[str, Any]] = None) -> Tuple[List[Dict[str, Any]], bool]:
    """
    Core function to process a list of orders with metadata.

    Args:
        order_list: List of order objects to process
        metadata_list: List of metadata objects corresponding to each order

    Returns:
        Tuple containing:
        - List of processed orders
        - Boolean indicating if transcript should be skipped
    """
    result = []
    skip_transcript = False

    # If metadata is None, create empty metadata for each order
    if metadata_list is None:
        metadata_list = [{}] * len(order_list)

    # Ensure lengths match
    assert len(order_list) == len(metadata_list), "Orders and metadata must have the same length."

    # Process each order
    for order, metadata in zip(order_list, metadata_list):
        order, should_skip_transcript, should_skip_order = process_order(order, metadata)
        if should_skip_transcript:
            skip_transcript = True
            break
        if should_skip_order:
            continue
        if order is not None:
            result.append(order)

    return result, skip_transcript


def parse_orders(order_list, metadata: Optional[Dict[str, Any]] = None) -> Tuple[List[Dict[str, Any]], bool]:
    """Parse orders."""
    metadata_list = [metadata or {}] * len(order_list)
    return process_multiple_orders(order_list, metadata_list)


def evaluate(
    output_dir: str,
    truth_file: Union[str, None] = None,
    pred_file: Union[str, None] = None
):
    """Evaluation pipeline."""

    # Load from files
    with open(truth_file, 'r') as f:
        truth_encounters = json.load(f)
    with open(pred_file, 'r') as f:
        pred_encounters = json.load(f)

    # check all keys in truth and pred matches
    if set(truth_encounters.keys()) != set(pred_encounters.keys()):
        raise ValueError("Truth and prediction keys do not match.")


    # Create a preprocessor config for both the manager and pairing matcher
    preprocessor_config = PreprocessorConfig(lowercase=True, remove_punctuation=True)

    # Initialize the manager with a basic configuration
    # In a real application, you might want to load this from a config file
    manager = EvaluationManager(
        output_directory=output_dir,
        fields={
            "description": MetricDict(metrics=["Match", "Strict", "Rouge1"]),
            "reason": MetricDict(metrics=["Rouge1"]),
            "order_type": MetricDict(metrics=["Strict"]),
            "provenance": MetricDict(metrics=["MultiLabel"]),
        },
        preprocessings={
            "description": True,
            "reason": True,
            "order_type": True,
            "order_level_metrics": True,
            "encounter_level_metrics": True
        },
        preprocessor_config=preprocessor_config,
        orders_metrics=MetricDict(
            metrics=["Rouge1"],
        ),
        encounter_metrics=MetricDict(
            metrics=["Rouge1"],
        )
    )

    # Initialize the pairing matcher with the preprocessor config
    pairing = PairingMatcher(
        output_directory=output_dir,
        preprocessing_config=preprocessor_config,
        field="description"  # Use description field for pairing
    )

    # Retrieve orders for each dialog, pair them and keep in accumulator.
    for idx, key in enumerate(truth_encounters):
        meta = {"transcript_id": key}
        truth_orders, skip_transcript = parse_orders(truth_encounters[key], meta)
        if skip_transcript:
            logger.warning("Skipping this transcript...")
            continue
        pred_orders, _ = parse_orders(pred_encounters[key], meta)

        logger.debug(f"********* {idx} *********")
        logger.debug(f"Pairing 1: {len(truth_orders)} : {truth_orders}")
        logger.debug(f"Pairing 2: {len(pred_orders)} : {pred_orders}")
        logger.debug(f"*************************")

        pairing(truth_orders, pred_orders)

    pairings = pairing.get_pairings(transpose=True)
    # Unpack the pairings tuple to match the new manager.process interface
    references, predictions, indices = pairings
    metrics = manager.process(references, predictions, indices)

    print(json.dumps(metrics, indent=4))

    reformatted_metrics = {}
    for K, V in metrics.items():
        for k, v in V.items():
            reformatted_metrics[K + "_" + k] = v

    if not os.path.exists(output_dir) and output_dir != "":
        os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "scores.json"), "w") as f:
        json.dump(reformatted_metrics, f, indent=4)

    # If output_dir empty string, no export. Else, ...
    # pairing.export(filename) # export pairings with match scores
    # manager.export(filename) # export metrics for each field


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate order extraction with simplified approach")
    parser.add_argument("-t", "--truth", type=str, help="Truth file")
    parser.add_argument("-p", "--pred", type=str, help="Prediction file")
    parser.add_argument("-o", "--output", type=str, default="test", help="Output directory path, default no output export")
    parser.add_argument("--debug", action="store_true", help="Set logging level to debug.")

    args = parser.parse_args()

    # Set logging level to debug if debug flag is set
    if args.debug:
        logger.setLevel(logging.DEBUG)

    evaluate(
        args.output,
        truth_file=args.truth,
        pred_file=args.pred
    )
