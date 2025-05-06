#!/usr/bin/env python
import os
import json
import argparse
import logging
from typing import List, Dict, Any, Optional, Generator, Tuple, Union

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from order import Order
from evaluation_simplified.pairing import PairingMatcher
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


def read_jsonl(file: str, metadata: Optional[Dict[str, Any]] = None) -> Tuple[List[Dict[str, Any]], bool]:
    """Load JSONL-like format from 'file' path."""
    with open(file, 'r') as raw:
        lines = raw.readlines()
    
    # Parse JSON objects from lines
    order_list = []
    for line in lines:
        if line.strip():
            order_list.append(json.loads(line))
    
    # Create metadata list (same metadata for each order)
    metadata_list = [metadata or {}] * len(order_list)
    
    # Use the core processing function
    return process_multiple_orders(order_list, metadata_list)


def input_generator(input_directory: str, left_ext: str = ".truth", right_ext: str = ".pred") -> Generator[Tuple[str, str], None, None]:
    """
    Instantiate a generator that provides the truth and pred paths along iterations.

    Args:
        - input_directory (str): path to directory containing *.truth and *.pred files for evaluation.
        - left_ext (str): extension to iterate on and to use on the left side (default '.truth').
        - right_ext (str): extension to look up and to use on the right side (default '.pred').

    Return:
        - Generator with on the left side a file path based on extention from 'left_ext', and on the right, 'right_ext'.
    """
    for i in os.listdir(input_directory):
        if i.endswith(left_ext):
            base_filename = i[:-len(left_ext)]
            truth_path = os.path.join(input_directory, f"{base_filename}{left_ext}")
            pred_path = os.path.join(input_directory, f"{base_filename}{right_ext}")
            yield truth_path, pred_path


def evaluate(
    output_dir: str,
    input_directory: Union[str, None] = None,
    truth_file: Union[str, None] = None,
    pred_file: Union[str, None] = None,
    use_input_dir_as_output: bool = False,
    truth_ext: str = ".truth",
    pred_ext: str = ".pred",
):
    """Evaluation pipeline."""

    # Load from directory or simple pair of files
    inputs = []
    if input_directory:
        inputs = input_generator(input_directory, left_ext=truth_ext, right_ext=pred_ext)
    elif truth_file and pred_file:
        inputs = [(truth_file, pred_file)]
    else:
        raise ValueError("'input_directory' or ('truth_file' and 'pred_file') must be provided to evaluate.")

    # Create a preprocessor config for both the manager and pairing matcher
    preprocessor_config = PreprocessorConfig(lowercase=True, remove_punctuation=True)
    
    # Initialize the manager with a basic configuration
    # In a real application, you might want to load this from a config file
    manager = EvaluationManager(
        output_directory=output_dir,
        fields={
            "description": MetricDict(metrics=["Rouge1"]),
            "reason": MetricDict(metrics=["Match", "Strict", "Rouge1"]),
            "order_type": MetricDict(metrics=["Match", "Strict", "Rouge1"]),
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
    for idx, (truth, pred) in enumerate(inputs):
        name = os.path.basename(truth)

        meta = {"transcript_id": name.removesuffix(truth_ext)}
        truth_orders, skip_transcript = read_jsonl(truth, meta)
        if skip_transcript:
            logger.warning("Skipping this transcript...")
            continue
        pred_orders, _ = read_jsonl(pred, meta)

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

    filename = "" # default auto-generated in objects
    if use_input_dir_as_output:
        filename = os.path.basename(input_directory)

    # If output_dir empty string, no export. Else, ...
    pairing.export(filename) # export pairings with match scores
    manager.export(filename) # export metrics for each field


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate order extraction with simplified approach")
    parser.add_argument("-i", "--input", type=str, help="Input directory path")
    parser.add_argument("-t", "--truth", type=str, help="Truth file")
    parser.add_argument("-p", "--pred", type=str, help="Prediction file")
    parser.add_argument("-te", "--truth_ext", type=str, default=".truth", help="Truth file extension with the dot.")
    parser.add_argument("-pe", "--pred_ext", type=str, default=".pred", help="Prediction file extension with the dot.")
    parser.add_argument("-o", "--output", type=str, default="", help="Output directory path, default no output export")
    parser.add_argument("-ef", "--export_filename_as_input_dir", action="store_true", help="Use input dir name as export filename")
    parser.add_argument("--debug", action="store_true", help="Set logging level to debug.")

    args = parser.parse_args()

    # Set logging level to debug if debug flag is set
    if args.debug:
        logger.setLevel(logging.DEBUG)

    evaluate(
        args.output,
        input_directory=args.input,
        truth_file=args.truth,
        pred_file=args.pred,
        use_input_dir_as_output=args.export_filename_as_input_dir,
        truth_ext=args.truth_ext,
        pred_ext=args.pred_ext,
    )
